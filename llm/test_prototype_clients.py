import unittest
from unittest.mock import patch
from llm.clients import RoleConfig,ModelClient,ModelResult
from llm.prototype_clients import PrototypeClient,_endpoints


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.config=RoleConfig('planner','vllm','qwen3.5-4b','http://127.0.0.1:8999/v1/chat/completions',enable_thinking=False)
        _endpoints.pop(self.config.api_url,None)

    def tearDown(self):_endpoints.pop(self.config.api_url,None)

    def test_two_failures_open_circuit_then_new_turn_auto_recovers(self):
        client=PrototypeClient(self.config)
        with patch('llm.prototype_clients.time.monotonic',return_value=10),patch('llm.prototype_clients.input_budget',side_effect=ConnectionError):
            self.assertEqual(client.call([]).error_code,'tokenizer_unavailable')
            self.assertEqual(client.call([]).error_code,'tokenizer_unavailable')
            self.assertEqual(client.call([]).error_code,'local_circuit_open')
        ok=ModelResult('{}','vllm','qwen3.5-4b',{},1,'ok')
        client.usage_unknown=True
        with patch('llm.prototype_clients.time.monotonic',return_value=41),patch('llm.prototype_clients.input_budget',return_value={'within_limit':True}),patch.object(ModelClient,'call',return_value=ok):
            self.assertTrue(client.call([]).ok)
            self.assertFalse(client.usage_unknown)
            self.assertEqual(_endpoints[self.config.api_url]['failures'],0)

    def test_oversized_prompt_does_not_call_generation(self):
        with patch('llm.prototype_clients.input_budget',return_value={'within_limit':False}),patch.object(ModelClient,'call') as call:
            result=PrototypeClient(self.config).call([])
        self.assertEqual(result.error_code,'serving_context_limit');call.assert_not_called()

    def test_wire_schema_removes_only_unsupported_uniqueness(self):
        from llm.prototype_clients import wire_schema
        from agent.qwen_interaction_policy import PLAN_SCHEMA
        from copy import deepcopy
        original=deepcopy(PLAN_SCHEMA)
        with patch('llm.prototype_clients.input_budget',return_value={'within_limit':True}),patch.object(ModelClient,'call',return_value=ModelResult('{}','vllm','qwen',{},0,'ok')) as call:
            PrototypeClient(self.config).call([],require_json=True,json_schema=PLAN_SCHEMA)
        self.assertEqual(PLAN_SCHEMA,original)
        self.assertNotIn('uniqueItems',call.call_args.kwargs['json_schema']['properties']['tool_requests'])
        self.assertEqual(wire_schema(PLAN_SCHEMA)['properties']['next_action'],PLAN_SCHEMA['properties']['next_action'])

    def test_cloud_failure_budget_is_not_reset_by_local_mixin(self):
        from dataclasses import replace
        cfg=replace(self.config,provider='deepseek',api_url='https://api.deepseek.com/chat/completions')
        client=PrototypeClient(cfg);client.usage_unknown=True
        with patch.object(ModelClient,'call',return_value=ModelResult('',cfg.provider,cfg.model_id,{},0,'budget_exceeded')),patch('llm.prototype_clients.input_budget') as tokenize:
            client.call([])
        self.assertTrue(client.usage_unknown);tokenize.assert_not_called()


if __name__=='__main__':unittest.main()
