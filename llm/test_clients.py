"""Contract tests for the Stage B role-scoped client; no paid API calls."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from llm.clients import ModelClient, RoleConfig, RunBudget, load_role_configs
from llm.llm import LLM, LLMCallError


def config(**changes):
    raw = {
        "provider": "vllm", "model_id": "served-qwen",
        "api_url": "http://127.0.0.1:8000/v1/chat/completions",
        "max_tokens": 100, "max_total_tokens": 10000,
    }
    raw.update(changes)
    return RoleConfig.from_mapping("extractor", raw)


class FakeResponse:
    status_code = 200

    def __init__(self, data=None, status=200):
        self.data = data or {
            "choices": [{"message": {"content": '{"next_action":"ask_followup_single"}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }
        self.status_code = status

    def json(self):
        return self.data


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response = response or FakeResponse()
        self.error = error
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error:
            raise self.error
        return self.response


MESSAGES = [{"role": "user", "content": "虚构测试输入"}]


class ModelClientTests(unittest.TestCase):
    def test_vllm_json_schema_is_sent_without_changing_legacy_requests(self):
        session = FakeSession()
        client = ModelClient(config(), session=session)
        schema = {'type': 'object', 'properties': {}, 'additionalProperties': False}
        client.call(MESSAGES, json_schema=schema)
        payload = session.calls[0][1]['json']
        self.assertEqual(payload['response_format']['type'], 'json_schema')
        self.assertEqual(payload['response_format']['json_schema']['schema'], schema)
        client.call(MESSAGES, require_json=True)
        self.assertEqual(session.calls[1][1]['json']['response_format'], {'type': 'json_object'})

    def test_config_rejects_plain_http_and_inline_secret(self):
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            config(api_url="http://example.com/v1/chat/completions")
        with self.assertRaisesRegex(ValueError, "secrets"):
            config(api_key="do-not-store")

    def test_loader_and_public_summary_have_no_secret(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roles.yaml"
            path.write_text(
                "roles:\n  planner:\n    provider: deepseek\n    model_id: test\n"
                "    api_url: https://example.com/v1/chat/completions\n"
                "    api_key_env: DEEPSEEK_API_KEY\n", encoding="utf-8"
            )
            cfg = load_role_configs(path)["planner"]
            self.assertEqual(cfg.provider, "deepseek")
            self.assertNotIn("api_key", cfg.public_summary())

    def test_success_records_usage_and_sends_configured_model(self):
        session = FakeSession()
        client = ModelClient(config(), session=session)
        result = client.call(MESSAGES, require_json=True)
        self.assertTrue(result.ok)
        self.assertEqual(result.usage["total_tokens"], 20)
        self.assertEqual(client.total_tokens, 20)
        self.assertEqual(session.calls[0][1]["json"]["model"], "served-qwen")
        self.assertNotIn("chat_template_kwargs", session.calls[0][1]["json"])

    def test_vllm_non_thinking_request_field(self):
        session = FakeSession()
        client = ModelClient(config(enable_thinking=False), session=session)
        self.assertTrue(client.call(MESSAGES).ok)
        self.assertEqual(session.calls[0][1]["json"]["chat_template_kwargs"], {"enable_thinking": False})
        with self.assertRaisesRegex(ValueError, "enable_thinking"):
            config(enable_thinking="false")

    def test_timeout_and_http_error_never_fallback_to_mock(self):
        timed = ModelClient(config(), session=FakeSession(error=requests.Timeout()))
        result = timed.call(MESSAGES)
        self.assertEqual(result.status, "timeout")
        self.assertFalse(result.fallback_used)
        failed = ModelClient(config(), session=FakeSession(response=FakeResponse(status=503)))
        self.assertEqual(failed.call(MESSAGES).error_code, "http_503")

    def test_json_and_truncation_are_visible_failures(self):
        bad_json = FakeResponse(data={
            "choices": [{"message": {"content": "not-json"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        })
        client = ModelClient(config(), session=FakeSession(response=bad_json))
        self.assertEqual(client.call(MESSAGES, require_json=True).status, "invalid_json")
        truncated = FakeResponse(data={
            "choices": [{"message": {"content": "{}"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4},
        })
        self.assertEqual(ModelClient(config(), session=FakeSession(response=truncated)).call(MESSAGES).status, "truncated")

    def test_malformed_provider_envelope_is_reported(self):
        client = ModelClient(config(), session=FakeSession(response=FakeResponse(data={"choices": ["wrong"]})))
        result = client.call(MESSAGES)
        self.assertEqual(result.status, "invalid_response")
        self.assertFalse(result.fallback_used)

    def test_missing_key_and_budget_stop_without_request(self):
        session = FakeSession()
        cloud = RoleConfig.from_mapping("planner", {
            "provider": "deepseek", "model_id": "test",
            "api_url": "https://example.com/v1/chat/completions",
            "api_key_env": "TEST_STAGE_B_MISSING_KEY",
        })
        with patch.dict("os.environ", {}, clear=True), patch("llm.clients.read_env_file", return_value={}):
            self.assertEqual(ModelClient(cloud, session=session).call(MESSAGES).status, "missing_key")
        self.assertEqual(len(session.calls), 0)
        client = ModelClient(config(max_total_tokens=20), session=session)
        self.assertEqual(client.call(MESSAGES).status, "budget_exceeded")
        self.assertEqual(len(session.calls), 0)

    def test_cloud_request_and_spending_reservation(self):
        cfg = RoleConfig.from_mapping("planner", {
            "provider": "deepseek", "model_id": "test", "api_url": "https://example.com/chat/completions",
            "api_key_env": "TEST_KEY", "enable_thinking": False, "max_tokens": 100,
            "input_cny_per_million": 9, "output_cny_per_million": 27,
        })
        session = FakeSession()
        with patch.dict("os.environ", {"TEST_KEY": "fake"}):
            blocked = ModelClient(cfg, session=session, budget=RunBudget(.000001))
            self.assertEqual(blocked.call(MESSAGES).status, "budget_exceeded")
            self.assertFalse(session.calls)
            budget = RunBudget(1)
            client = ModelClient(cfg, session=session, budget=budget)
            result = client.call(MESSAGES, require_json=True)
        self.assertTrue(result.ok)
        self.assertAlmostEqual(budget.accounted_cny, .00036)
        payload = session.calls[0][1]["json"]
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_unknown_usage_stops_following_paid_calls(self):
        session = FakeSession(response=FakeResponse(data={
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        }))
        client = ModelClient(config(), session=session)
        self.assertEqual(client.call(MESSAGES).status, "invalid_response")
        self.assertEqual(client.call(MESSAGES).status, "budget_exceeded")
        self.assertEqual(len(session.calls), 1)


class LegacyStrictModeTests(unittest.TestCase):
    def test_missing_real_provider_raises_without_mock(self):
        llm = LLM(provider="deepseek", strict=True)
        with patch.object(llm, "is_deepseek_configured", return_value=False):
            with self.assertRaises(LLMCallError):
                llm.call("json")
        self.assertEqual(llm.get_metrics()["mock_fallback_count"], 0)
        self.assertEqual(llm.get_metrics()["error_count"], 1)

    def test_provider_failure_is_counted_and_not_mocked(self):
        llm = LLM(provider="deepseek", strict=True)
        with patch.object(llm, "is_deepseek_configured", return_value=True), patch.object(
            llm, "_deepseek_call", side_effect=requests.Timeout()
        ):
            with self.assertRaises(LLMCallError):
                llm.call("json")
        self.assertEqual(llm.get_metrics()["call_count"], 1)
        self.assertEqual(llm.get_metrics()["mock_fallback_count"], 0)
        self.assertEqual(llm.last_provider_used, "deepseek")


if __name__ == "__main__":
    unittest.main()
