# 统一调用大模型接口，支持 DeepSeek 和 Mock 两种模式
# fallback到mock（无接口时可运行）
# 支持JSON抽取（从不规范LLM输出中恢复结构）
import json
import os
import re

from config_manager import apply_config_to_environment

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
        }

    def call(self, prompt):
        self.last_error = ""

        if self.provider in ("deepseek", "auto") and self.is_deepseek_configured():
            try:
                result = self._deepseek_call(prompt)
                self.last_provider_used = "deepseek"
                return result
            except Exception as exc:
                self.last_error = f"DeepSeek call failed: {exc}"

        self.last_provider_used = "mock"
        if not self.last_error:
            self.last_error = "DeepSeek is unavailable, fallback to mock."
        return self._mock_call(prompt)

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
        return self._extract_text(response.json())

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
