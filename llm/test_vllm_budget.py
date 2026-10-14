import unittest
from unittest.mock import Mock,patch

from llm.clients import RoleConfig
from llm.vllm_budget import input_budget


class VllmBudgetTests(unittest.TestCase):
    def test_exact_template_boundary_and_no_text_truncation(self):
        config=RoleConfig('responder','vllm','qwen3.5-4b','http://127.0.0.1:8000/v1/chat/completions',enable_thinking=False)
        messages=[{'role':'user','content':'咳嗽'*2000}]
        response=Mock();response.json.return_value={'tokens':list(range(3584))}
        with patch('llm.vllm_budget.requests.post',return_value=response) as post:
            result=input_budget(config,messages,512)
            self.assertTrue(result['within_limit'])
            self.assertEqual(post.call_args.kwargs['json']['messages'],messages)
            self.assertEqual(post.call_args.kwargs['json']['chat_template_kwargs'],{'enable_thinking':False})
            self.assertTrue(post.call_args.kwargs['json']['add_generation_prompt'])
            self.assertFalse(input_budget(config,messages,513)['within_limit'])

    def test_invalid_token_list_is_not_a_zero_length_success(self):
        config=RoleConfig('responder','vllm','qwen3.5-4b','http://127.0.0.1:8000/v1/chat/completions',enable_thinking=False)
        response=Mock();response.json.return_value={'tokens':['not an id']}
        with patch('llm.vllm_budget.requests.post',return_value=response),self.assertRaises(ValueError):
            input_budget(config,[],512)

    def test_cloud_or_thinking_configuration_cannot_use_local_measurement(self):
        with patch('llm.vllm_budget.requests.post') as post:
            with self.assertRaises(ValueError):
                input_budget(RoleConfig('responder','deepseek','foo','https://api.deepseek.com/chat/completions'),[],512)
            self.assertFalse(post.called)


if __name__=='__main__':unittest.main()
