import unittest
from unittest.mock import patch
from agent.delivery_recovery import build
from agent.test_semantic_runtime import SemanticRuntimeTests,delta
from training.test_fact_contract import fact


class RecoveryTests(unittest.TestCase):
    setUp=SemanticRuntimeTests.setUp
    def agent(self,*outputs,**kwargs):
        agent,client=SemanticRuntimeTests.agent(self,*outputs,**kwargs)
        agent.fact_extractor.turn_requests_version='delivery_turn_requests_v1'
        return agent,client

    def test_omitted_deferred_input_is_a_question_not_a_fact_or_low_risk_claim(self):
        agent,_=self.agent(delta(fact('鼻塞')),delta())
        agent.run('鼻塞。');records=agent.get_case_snapshot()['fact_ledger']['records']
        response=agent.run('心悸的事以后再谈。')
        state=agent.get_case_snapshot()
        self.assertIn('你现在有心悸吗',response)
        self.assertEqual(state['fact_ledger']['records'],records)
        self.assertEqual(state['delivery_recovery']['kind'],'missing_deferred_candidate')
        self.assertNotIn('心悸',state['symptoms'])

    def test_questions_do_not_create_facts_from_hypotheses_other_people_or_known_problems(self):
        for text in ('如果以后有耳痛，以后再说','姐姐耳痛的事以后再谈','“耳痛的事以后再谈”'):
            self.assertEqual(build({},text)['kind'],'none')
        self.assertEqual(build({'fact_view':{'problems':[{'value':'心悸'}]}},'心悸的事以后再谈')['kind'],'none')

    def test_root_specific_questions_separate_service_denial_context_and_hypothesis(self):
        for error,phrase in [('provider:network_error','模型服务'),('explicit_current_denial_missing_from_facts','否认'),
                             ('occurrence_context_is_not_elapsed_duration','情境'),
                             ('literal_positive_condition_missing_scoped_fact','假设'),
                             ('delivery_red_flag_without_concept_evidence:f2','没有原文依据')]:
            self.assertIn(phrase,build({'fact_extraction_error':error},'患者原文')['question'])


if __name__=='__main__':unittest.main()
