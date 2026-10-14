import unittest
from copy import deepcopy
from agent.semantic_extraction import SemanticExtractor
from agent.test_semantic_runtime import FixedClient
from llm.input_boundary import schema_order_messages
from llm.semantic_prompt import augment,INSTRUCTION
from memory.memory import ConversationMemory


class SemanticPromptTests(unittest.TestCase):
    def test_locked_prompt_and_user_bytes_stay_unchanged(self):
        m=ConversationMemory();m.enable_fact_mode()
        original=schema_order_messages(m.fact_input('暂时没有新的资料。'));saved=deepcopy(original)
        self.assertEqual(augment(original,'locked'),original)
        changed=augment(original,'coverage_v1')
        self.assertEqual(original,saved);self.assertEqual(changed[1],original[1])
        self.assertEqual(changed[0]['content'],original[0]['content']+'\n'+INSTRUCTION)

    def test_actual_extractor_optin_passes_only_shared_instructions(self):
        class Capturing(FixedClient):
            def call(self,messages,**kwargs):self.messages=messages;return super().call(messages,**kwargs)
        m=ConversationMemory();m.enable_fact_mode();client=Capturing({'facts':[]})
        audit=SemanticExtractor(client,boundary_version='v4',prompt_version='coverage_v1').run(m,'暂时没有新的资料。')
        self.assertEqual(audit['status'],'ok');self.assertIn(INSTRUCTION,client.messages[0]['content'])
        self.assertEqual(audit['prompt_version'],'coverage_v1')


if __name__=='__main__':unittest.main()
