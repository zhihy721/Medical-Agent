"""Portable safety regression with synthetic inputs and deterministic model outputs."""
from copy import deepcopy
import unittest
from agent.factory import create_agent
from agent.semantic_extraction import SemanticExtractor
from agent.test_semantic_runtime import FixedClient,delta
from llm.fact_contract_runtime import V2
from llm.fact_projection_policy import VERSION
from llm.delivery_polarity_guard import check
from memory.memory import ConversationMemory
from training.test_fact_contract import fact


def extractor(client):
    ex=SemanticExtractor(client,contract_version=V2,representation_mode='finite_v5',boundary_version='v1',prompt_version='g6_v7',recovery_mode='checked_v16')
    ex.delivery_guard='delivery_grounding_guard_v1'
    ex.input_safety_version='literal_breathing_safety_v2'
    return ex


class PublicSafetyTests(unittest.TestCase):
    def test_current_denial_cannot_commit_model_positive_or_erase_old_state(self):
        ds=[delta(fact('呼吸困难','没有呼吸困难',field='red_flags',polarity='negated')),
            delta(fact('呼吸困难','现在也还是没有呼吸困难',field='red_flags'))]
        memory=ConversationMemory();memory.enable_fact_mode(V2,representation_policy=VERSION)
        actor=create_agent(None,memory,runtime='langgraph',fact_extractor=extractor(FixedClient(*ds)),response_mode='template')
        actor.run('没有呼吸困难。');before=deepcopy(memory.case_state['fact_ledger']['records']);before_risk=memory.case_state['risk_level']
        actor.run('现在也还是没有呼吸困难。')
        self.assertEqual(actor.get_graph_state().values['extraction_trace']['status'],'error')
        self.assertEqual(memory.case_state['risk_level'],before_risk)
        self.assertNotEqual(memory.case_state['risk_level'],'HIGH')
        self.assertEqual(memory.case_state['fact_ledger']['records'],before)

    def test_unambiguous_denials_reject_positive_values(self):
        for text,value in [('目前没有喘不过气','呼吸困难'),('明确否认头痛','头痛'),('我没有任何便血','便血')]:
            with self.subTest(text=text),self.assertRaisesRegex(ValueError,'explicit_current_denial_conflicts'):
                check(dict(current_turn=text,known_facts=[]),delta(fact(value,text)))

    def test_other_person_denial_does_not_block_patient_high_risk(self):
        text='姐姐没有呼吸困难，我现在呼吸困难。'
        d=delta(fact('呼吸困难','我现在呼吸困难',field='red_flags'))
        memory=ConversationMemory();memory.enable_fact_mode(V2,representation_policy=VERSION)
        actor=create_agent(None,memory,runtime='langgraph',fact_extractor=extractor(FixedClient(d)),response_mode='template')
        actor.run(text)
        self.assertEqual(memory.case_state['risk_level'],'HIGH')

    def test_existing_real_high_is_not_erased_by_rejected_transaction(self):
        ds=[delta(fact('呼吸困难',field='red_flags')),
            delta(fact('呼吸困难','现在没有呼吸困难',field='red_flags'))]
        memory=ConversationMemory();memory.enable_fact_mode(V2,representation_policy=VERSION)
        actor=create_agent(None,memory,runtime='langgraph',fact_extractor=extractor(FixedClient(*ds)),response_mode='template')
        actor.run('呼吸困难。');before=deepcopy(memory.case_state['fact_ledger']['records'])
        actor.run('现在没有呼吸困难。')
        self.assertEqual(memory.case_state['risk_level'],'HIGH')
        self.assertEqual(memory.case_state['fact_ledger']['records'],before)


if __name__=='__main__':unittest.main()
