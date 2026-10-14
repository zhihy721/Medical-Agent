"""Role-scoped OpenAI-compatible model calls for Stage B.

A failed request returns an explicit status. This client never substitutes Mock text.
Provider-specific request options are intentionally deferred until compatibility probes.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from threading import Lock
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml

from config_manager import read_env_file


ROLES = frozenset({"extractor", "intent", "planner", "reviewer", "responder", "judge", "teacher"})
PROVIDERS = frozenset({"deepseek", "bailian", "vllm"})
ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


@dataclass(frozen=True)
class RoleConfig:
    role: str
    provider: str
    model_id: str
    api_url: str
    api_key_env: str = ""
    timeout_s: float = 30.0
    max_tokens: int = 512
    temperature: float = 0.0
    max_total_tokens: int = 0
    enable_thinking: bool | None = None
    input_cny_per_million: float = 0.0
    output_cny_per_million: float = 0.0
    cached_input_cny_per_million: float = 0.0

    @classmethod
    def from_mapping(cls, role: str, raw: dict) -> "RoleConfig":
        if not isinstance(raw, dict):
            raise ValueError(f"{role}: config must be a mapping")
        if any(key in raw for key in ("api_key", "token", "secret")):
            raise ValueError(f"{role}: secrets must not be stored in role config")
        try:
            config = cls(
                role=role,
                provider=str(raw["provider"]).strip().lower(),
                model_id=str(raw["model_id"]).strip(),
                api_url=str(raw["api_url"]).strip(),
                api_key_env=str(raw.get("api_key_env") or "").strip(),
                timeout_s=float(raw.get("timeout_s", 30)),
                max_tokens=int(raw.get("max_tokens", 512)),
                temperature=float(raw.get("temperature", 0)),
                max_total_tokens=int(raw.get("max_total_tokens", 0)),
                enable_thinking=raw.get("enable_thinking"),
                input_cny_per_million=float(raw.get("input_cny_per_million", 0)),
                output_cny_per_million=float(raw.get("output_cny_per_million", 0)),
                cached_input_cny_per_million=float(raw.get("cached_input_cny_per_million", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{role}: invalid role config") from exc
        if role not in ROLES or config.provider not in PROVIDERS or not config.model_id:
            raise ValueError(f"{role}: unsupported role/provider or empty model_id")
        url = urlparse(config.api_url)
        loopback = url.hostname in {"127.0.0.1", "localhost", "::1"}
        if url.scheme != "https" and not (url.scheme == "http" and loopback):
            raise ValueError(f"{role}: use HTTPS or a loopback SSH tunnel")
        if not url.hostname or not url.path.endswith("/chat/completions") or url.username or url.password or url.query or url.fragment:
            raise ValueError(f"{role}: api_url must be a plain Chat Completions endpoint")
        if config.api_key_env and not ENV_NAME.fullmatch(config.api_key_env):
            raise ValueError(f"{role}: api_key_env must name an environment variable")
        if config.provider != "vllm" and not config.api_key_env:
            raise ValueError(f"{role}: cloud provider needs api_key_env")
        if not 0 < config.timeout_s <= 120 or not 0 < config.max_tokens <= 8192:
            raise ValueError(f"{role}: timeout_s/max_tokens out of range")
        if not 0 <= config.temperature <= 2 or config.max_total_tokens < 0:
            raise ValueError(f"{role}: temperature/token budget out of range")
        if config.enable_thinking is not None and not isinstance(config.enable_thinking, bool):
            raise ValueError(f"{role}: enable_thinking must be a boolean")
        prices = (config.input_cny_per_million, config.output_cny_per_million, config.cached_input_cny_per_million)
        if any(not 0 <= price < 10000 for price in prices):
            raise ValueError(f"{role}: invalid price")
        return config

    def public_summary(self) -> dict:
        return asdict(self)


@dataclass
class RunBudget:
    """Shared sequential-run spending guard. Unknown charges retain the reservation."""
    limit_cny: float
    accounted_cny: float = 0.0
    stopped: bool = False
    _lock: Lock = field(default_factory=Lock, repr=False)

    def reserve(self, amount: float) -> bool:
        with self._lock:
            if self.stopped or amount < 0 or self.accounted_cny + amount > self.limit_cny:
                return False
            self.accounted_cny += amount
            return True

    def settle(self, reserved: float, actual: float | None) -> None:
        with self._lock:
            if actual is None:
                self.stopped = True
            else:
                self.accounted_cny += actual - reserved
                if actual > reserved or self.accounted_cny > self.limit_cny:
                    self.stopped = True


@dataclass(frozen=True)
class ModelResult:
    content: str
    provider: str
    model_id: str
    usage: dict
    latency_ms: float
    status: str
    error_code: str = ""
    fallback_used: bool = False
    estimated_cost_cny: float | None = None
    budget_accounted_cny: float = 0.0
    returned_model_id: str = ""
    system_fingerprint: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def load_role_configs(path: str | Path) -> dict[str, RoleConfig]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"roles"} or not isinstance(raw["roles"], dict):
        raise ValueError("role config must contain only a roles mapping")
    unknown = set(raw["roles"]) - ROLES
    if unknown:
        raise ValueError(f"unsupported roles: {sorted(unknown)}")
    return {role: RoleConfig.from_mapping(role, value) for role, value in raw["roles"].items()}


class ModelClient:
    def __init__(self, config: RoleConfig, *, session=None, budget: RunBudget | None = None):
        self.config = config
        self.session = session or requests.Session()
        self.total_tokens = 0
        self.budget = budget
        self.usage_unknown = False

    def _api_key(self) -> str:
        if not self.config.api_key_env:
            return ""
        return (os.getenv(self.config.api_key_env) or read_env_file().get(self.config.api_key_env) or "").strip()

    def call(self, messages: list[dict], *, require_json: bool = False, json_schema: dict | None = None) -> ModelResult:
        cfg = self.config
        started = time.perf_counter()
        estimated_cost = None
        accounted_cost = 0.0
        returned_model = fingerprint = ""

        def result(status: str, *, content: str = "", usage: dict | None = None, error_code: str = "") -> ModelResult:
            return ModelResult(content, cfg.provider, cfg.model_id, usage or {},
                               round((time.perf_counter() - started) * 1000, 1), status, error_code,
                               estimated_cost_cny=estimated_cost, budget_accounted_cny=accounted_cost,
                               returned_model_id=returned_model, system_fingerprint=fingerprint)

        if not isinstance(messages, list) or not messages or any(
            not isinstance(item, dict) or item.get("role") not in {"system", "user", "assistant", "tool"}
            or not isinstance(item.get("content"), str) for item in messages
        ):
            return result("invalid_request", error_code="messages")
        if json_schema is not None:
            if cfg.provider != 'vllm' or not isinstance(json_schema, dict):
                return result('invalid_request', error_code='schema_constraint_requires_vllm')
            require_json = True
        key = self._api_key()
        if cfg.api_key_env and not key:
            return result("missing_key", error_code=cfg.api_key_env)
        # UTF-8 bytes plus generous chat framing is a conservative reservation,
        # not a tokenizer count. Unknown usage stops paid runs instead of guessing.
        prompt_bound = 128 + sum(len(m["content"].encode("utf-8")) + 64 for m in messages)
        token_reservation = prompt_bound + cfg.max_tokens
        if self.usage_unknown or (cfg.max_total_tokens and self.total_tokens + token_reservation > cfg.max_total_tokens):
            return result("budget_exceeded", error_code="token_reservation")
        reserved_cost = (prompt_bound * cfg.input_cny_per_million + cfg.max_tokens * cfg.output_cny_per_million) / 1e6
        if self.budget is not None and cfg.provider != "vllm":
            if cfg.input_cny_per_million <= 0 or cfg.output_cny_per_million <= 0:
                return result("invalid_request", error_code="missing_price")
            if not self.budget.reserve(reserved_cost):
                return result("budget_exceeded", error_code="cny_reservation")
            accounted_cost = reserved_cost

        def unknown_charge():
            self.total_tokens += token_reservation
            self.usage_unknown = True
            if self.budget is not None and cfg.provider != "vllm":
                self.budget.settle(reserved_cost, None)
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        payload = {
            "model": cfg.model_id, "messages": messages, "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
        if cfg.enable_thinking is not None:
            if cfg.provider == "deepseek":
                payload["thinking"] = {"type": "enabled" if cfg.enable_thinking else "disabled"}
            elif cfg.provider == "bailian":
                payload["enable_thinking"] = cfg.enable_thinking
            elif cfg.provider == "bailian":
                payload["enable_thinking"] = cfg.enable_thinking
            else:
                payload["chat_template_kwargs"] = {"enable_thinking": cfg.enable_thinking}
        if require_json:
            payload["response_format"] = ({"type": "json_schema", "json_schema": {
                "name": "extraction", "schema": json_schema, "strict": True}}
                if json_schema is not None else {"type": "json_object"})
        try:
            response = self.session.post(cfg.api_url, headers=headers, json=payload, timeout=cfg.timeout_s)
        except requests.Timeout:
            unknown_charge()
            return result("timeout", error_code="request_timeout")
        except requests.RequestException:
            unknown_charge()
            return result("network_error", error_code="request_failed")
        if response.status_code >= 400:
            unknown_charge()
            return result("http_error", error_code=f"http_{response.status_code}")
        try:
            data = response.json()
            if not isinstance(data, dict) or not isinstance(data.get("choices"), list):
                raise ValueError("invalid response envelope")
            choice = data["choices"][0]
            if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
                raise ValueError("invalid message")
            content = choice["message"].get("content") or ""
            raw_usage = data.get("usage")
            if not isinstance(raw_usage, dict) or not {"prompt_tokens", "completion_tokens"} <= set(raw_usage):
                raise ValueError("invalid usage")
            usage = {
                name: int(raw_usage.get(name) or 0)
                for name in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
            if any(value < 0 for value in usage.values()):
                raise ValueError("negative usage")
            usage["total_tokens"] = usage["total_tokens"] or usage["prompt_tokens"] + usage["completion_tokens"]
            usage["prompt_cache_hit_tokens"] = int(raw_usage.get("prompt_cache_hit_tokens") or (raw_usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
            if not 0 <= usage["prompt_cache_hit_tokens"] <= usage["prompt_tokens"]:
                raise ValueError("invalid cached usage")
            returned_model = str(data.get("model") or "")
            fingerprint = str(data.get("system_fingerprint") or "")
        except (KeyError, IndexError, TypeError, ValueError, OverflowError):
            unknown_charge()
            return result("invalid_response", error_code="response_shape")
        self.total_tokens += usage["total_tokens"] or usage["prompt_tokens"] + usage["completion_tokens"]
        if cfg.provider == "vllm":
            estimated_cost = 0.0  # API fees only; electricity/hardware are not measured here.
        elif cfg.input_cny_per_million > 0 and cfg.output_cny_per_million > 0:
            hits = usage["prompt_cache_hit_tokens"]
            cache_rate = cfg.cached_input_cny_per_million or cfg.input_cny_per_million
            estimated_cost = ((usage["prompt_tokens"] - hits) * cfg.input_cny_per_million + hits * cache_rate + usage["completion_tokens"] * cfg.output_cny_per_million) / 1e6
        if self.budget is not None and cfg.provider != "vllm":
            self.budget.settle(reserved_cost, estimated_cost)
            accounted_cost = estimated_cost if estimated_cost is not None else reserved_cost
        if usage["total_tokens"] > token_reservation:
            self.usage_unknown = True
            if self.budget is not None:
                self.budget.stopped = True
            return result("budget_exceeded", content=content, usage=usage, error_code="reservation_underestimated")
        if choice.get("finish_reason") == "length":
            return result("truncated", content=content, usage=usage, error_code="max_tokens")
        if choice.get("finish_reason") != "stop" or not isinstance(content, str) or not content.strip():
            return result("invalid_response", usage=usage, error_code="incomplete_content")
        if require_json:
            try:
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise ValueError("JSON object required")
            except (json.JSONDecodeError, ValueError):
                return result("invalid_json", content=content, usage=usage, error_code="json_object")
        return result("ok", content=content, usage=usage)
