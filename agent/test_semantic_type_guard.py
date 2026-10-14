"""Real extraction transactions for unsupported normal-state and typed values."""
from copy import deepcopy
import unittest
from agent.semantic_extraction import SemanticExtractor
from agent.test_semantic_runtime import FixedClient,delta
from training.test_fact_contract import fact
from memory.memory import ConversationMemory


class SemanticTypeGuardTests(unittest.TestCase):
    guard='guarded_v9'

    def run_delta(self,text,proposal):
        m=ConversationMemory();m.enable_fact_mode()
        m.apply_fact_delta(m.fact_input('耳痛三天。'),delta(fact('耳痛'),fact('三天',id='f2',field='duration',about='f1')))
        before=deepcopy(m.case_state['fact_ledger']['records'])
        audit=SemanticExtractor(FixedClient(proposal,proposal),recovery_mode=self.guard).run(m,text)
        return m,before,audit

    def test_normal_statement_does_not_support_positive_symptom(self):
        m,before,audit=self.run_delta('寒热感觉和平时一样。',delta(fact('发热','寒热感觉和平时一样')))
        self.assertEqual(audit['status'],'error');self.assertEqual(m.case_state['fact_ledger']['records'],before)

    def test_same_normal_statement_does_not_support_cold_heat_abnormal_value(self):
        m,before,audit=self.run_delta('寒热感觉和平时一样。',delta(fact('发热','寒热感觉和平时一样',field='cold_heat')))
        self.assertEqual(audit['status'],'error');self.assertEqual(m.case_state['fact_ledger']['records'],before)

    def test_problem_name_is_not_a_severity_value(self):
        m,before,audit=self.run_delta('耳朵的程度是轻微。',delta(fact('耳痛','耳朵的程度是轻微',field='severity',about='t1:f1')))
        self.assertEqual(audit['status'],'error');self.assertEqual(m.case_state['fact_ledger']['records'],before)

    def test_literal_state_and_explicit_abnormality_are_not_rewritten(self):
        for text,proposal in [('寒热感觉和平时一样。',delta(fact('和平时一样','寒热感觉和平时一样',field='cold_heat'))),
            ('饮水正常，但现在有发热。',delta(fact('发热','现在有发热')))]:
            with self.subTest(text=text):self.assertEqual(self.run_delta(text,proposal)[2]['status'],'ok')


if __name__=='__main__':unittest.main()
