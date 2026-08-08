# 统一调用大模型接口，支持 DeepSeek 和 Mock 两种模式
# fallback到mock（无接口时可运行）
# 支持JSON抽取（从不规范LLM输出中恢复结构）
# 内置可观测性埋点：每次调用记录耗时、provider、token 用量，并累计 metrics
import json
import os
import re
import time

from config_manager import apply_config_to_environment
from observability.events import event_logger
from observability.logger import get_logger

_logger = get_logger("llm")

try:
    import requests
except ImportError:
    requests = None


class LLM:
    def __init__(self, system_prompt=None, provider=None):
        self.system_prompt = system_prompt or ""
        self.provider = (provider or os.getenv("LLM_PROVIDER", "deepseek")).lower()
        self.last_provider_used = "mock"
        self.last_error = ""
        # 指标累计器：调用次数、耗时、token 用量、降级与错误次数
        self._metrics = {
            "call_count": 0,
            "error_count": 0,
            "mock_fallback_count": 0,
            "total_latency_ms": 0.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
        }
        # 最近一次 DeepSeek 响应中的 usage 字段
        self._last_usage = {}
        self._load_config()

    def _load_config(self):
        apply_config_to_environment()

    def is_deepseek_configured(self):
        return bool(os.getenv("DEEPSEEK_API_URL") and os.getenv("DEEPSEEK_API_KEY"))

    def get_runtime_status(self):
        return {
            "configured_provider": self.provider,
            "deepseek_configured": self.is_deepseek_configured(),
            "last_provider_used": self.last_provider_used,
            "last_error": self.last_error,
            "metrics": self.get_metrics(),
        }

    def get_metrics(self):
        """返回累计指标摘要，含平均延迟与 mock 降级率。"""
        metrics = dict(self._metrics)
        call_count = metrics["call_count"]
        metrics["avg_latency_ms"] = round(metrics["total_latency_ms"] / call_count, 1) if call_count else 0.0
        metrics["mock_fallback_rate"] = round(metrics["mock_fallback_count"] / call_count, 3) if call_count else 0.0
        metrics["total_latency_ms"] = round(metrics["total_latency_ms"], 1)
        return metrics

    def call(self, prompt):
        self.last_error = ""
        started = time.perf_counter()
        fallback = False

        if self.provider in ("deepseek", "auto") and self.is_deepseek_configured():
            try:
                result = self._deepseek_call(prompt)
                self.last_provider_used = "deepseek"
                self._record_call(started, fallback)
                return result
            except Exception as exc:
                self.last_error = f"DeepSeek call failed: {exc}"
                _logger.warning("DeepSeek call failed, fallback to mock: %s", exc)

        self.last_provider_used = "mock"
        fallback = True
        if not self.last_error:
            self.last_error = "DeepSeek is unavailable, fallback to mock."
        result = self._mock_call(prompt)
        self._record_call(started, fallback)
        return result

    def _record_call(self, started, fallback):
        """统一埋点：累计 metrics、写日志、emit llm_call 事件。"""
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        usage = self._last_usage if self.last_provider_used == "deepseek" else {}
        self._metrics["call_count"] += 1
        self._metrics["total_latency_ms"] += elapsed_ms
        if fallback:
            self._metrics["mock_fallback_count"] += 1
        if self.last_error:
            self._metrics["error_count"] += 1
        self._metrics["total_prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        self._metrics["total_completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)

        _logger.info(
            "LLM call provider=%s elapsed_ms=%s fallback=%s error=%s",
            self.last_provider_used,
            elapsed_ms,
            fallback,
            bool(self.last_error),
        )
        event_logger.emit(
            "llm_call",
            provider=self.last_provider_used,
            elapsed_ms=elapsed_ms,
            fallback=fallback,
            error=self.last_error,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        )

    def extract_json(self, prompt):
        text = self.call(prompt)
        return self._parse_json_from_text(text)

    def _deepseek_call(self, prompt):
        if not requests:
            raise RuntimeError("requests is required for DeepSeek calls in this project")

        api_url = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
        payload = {
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": int(os.getenv("DEEPSEEK_MAX_TOKENS", 512)),
            "temperature": float(os.getenv("DEEPSEEK_TEMPERATURE", 0.2)),
        }
        headers = {
            "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
            "Content-Type": "application/json",
        }

        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        # 记录 token 用量，供 metrics 与事件流使用
        self._last_usage = (data.get("usage") or {}) if isinstance(data, dict) else {}
        return self._extract_text(data)

    def _extract_text(self, data):
        if isinstance(data, dict) and data.get("choices"):
            choice = data["choices"][0]
            message = choice.get("message", {})
            return message.get("content", "").strip()
        return json.dumps(data, ensure_ascii=False)

    def _parse_json_from_text(self, text):
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return {}
        return {}

    def _mock_call(self, prompt):
        lowered = prompt.lower()
        if "只输出 json" in prompt or "json" in lowered:
            return "{}"
        if "追问" in prompt:
            return "为了继续判断，请补充症状持续时间或严重程度。"
        if "最终回复" in prompt or "风险等级" in prompt:
            return "根据目前信息，建议继续观察并在症状加重时及时就医。以上仅供分诊参考，不能替代医生面诊。"
        return "当前处于演示模式，未调用外部大模型接口。"
