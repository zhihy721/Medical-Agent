import os
import unittest
from unittest.mock import patch
from agent.policy_config import policy_from_environment, extraction_from_environment
from llm.clients import RoleConfig


class PolicyEnvironmentTests(unittest.TestCase):
    def test_model_responder_token_budget_opt_in_keeps_other_roles_unmodified(self):
        from llm.vllm_budget import BudgetedModelClient
        roles={name:RoleConfig(name,'vllm','extractor-adapter' if name=='extractor' else 'base',
            'http://127.0.0.1:8000/v1/chat/completions',enable_thinking=False) for name in ('extractor','intent','responder')}
        with patch.dict(os.environ,{'AGENT_EXTRACTOR_MODE':'semantic','AGENT_RESPONDER_MODE':'model',
            'AGENT_RESPONDER_BUDGET_MODE':'serving_tokenizer'},clear=True),patch('agent.policy_config.load_role_configs',return_value=roles):
            config=extraction_from_environment()
            self.assertIsInstance(config['responder'],BudgetedModelClient)
            self.assertEqual(config['responder'].config.model_id,'base')
            self.assertEqual(config['fact_extractor'].client.config.model_id,'extractor-adapter')
            self.assertFalse(hasattr(config['fact_extractor'].client,'input_budget'))
    def test_versioned_prompt_can_be_selected_without_changing_legacy_default(self):
        with patch.dict(os.environ,{},clear=True):self.assertIsNone(policy_from_environment())
        for version in ('v2','v3'):
            with patch.dict(os.environ,{'AGENT_PLANNER_MODE':'rule','AGENT_REVIEWER_MODE':'rule','AGENT_POLICY_PROMPT_VERSION':version},clear=True):
                policy=policy_from_environment()
                self.assertEqual(policy.prompt_version,version)
                self.assertIsNone(policy.planner_client)
                self.assertIsNone(policy.reviewer_client)

    def test_unknown_prompt_is_rejected_before_any_model_call(self):
        with patch.dict(os.environ,{'AGENT_PLANNER_MODE':'rule','AGENT_REVIEWER_MODE':'rule','AGENT_POLICY_PROMPT_VERSION':'v99'},clear=True):
            with self.assertRaises(ValueError):policy_from_environment()

    def test_extraction_opt_ins_are_explicit_and_role_isolated(self):
        roles = {name: RoleConfig(name, 'vllm', 'extractor-adapter' if name == 'extractor' else 'base',
                 'http://127.0.0.1:8000/v1/chat/completions') for name in ('extractor', 'intent', 'responder')}
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(extraction_from_environment(), {})
        with patch.dict(os.environ, {'AGENT_EXTRACTOR_MODE': 'semantic', 'AGENT_INTENT_MODE': 'model',
                'AGENT_INTENT_PROMPT_VERSION': 'v2', 'AGENT_EXTRACTOR_BOUNDARY_VERSION': 'v2',
                'AGENT_EXTRACTOR_RECOVERY_MODE': 'guarded_v1'}, clear=True), \
                patch('agent.policy_config.load_role_configs', return_value=roles):
            config = extraction_from_environment()
            extractor = config['fact_extractor']
            self.assertEqual(extractor.client.config.model_id, 'extractor-adapter')
            self.assertEqual(extractor.intent.client.config.model_id, 'base')
            self.assertEqual(extractor.intent.prompt_version, 'v2')
            self.assertEqual(extractor.boundary_version, 'v2')
            self.assertEqual(extractor.recovery_mode, 'guarded_v1')
            self.assertEqual(config['response_mode'], 'template')

    def test_bad_extraction_mode_rejected_without_provider_call(self):
        from agent.semantic_extraction import SemanticExtractor
        for kwargs in ({'recovery_mode': 'infinite'}, {'boundary_version': 'v99'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SemanticExtractor(None, **kwargs)


if __name__=='__main__':unittest.main()
