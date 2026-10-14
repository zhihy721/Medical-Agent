"""Replay completed development failures at the real extraction transaction."""
from copy import deepcopy
import unittest
from agent.semantic_extraction import SemanticExtractor
from agent.test_semantic_runtime import FixedClient,delta
from memory.memory import ConversationMemory
from training.test_fact_contract import fact


def memory():
    m=ConversationMemory();m.enable_fact_mode()
    m.apply_fact_delta(m.fact_input('口渴两天。'),delta(fact('口渴'),fact('两天',id='f2',field='duration',about='f1')))
    return m


class AttributeTransactionTests(unittest.TestCase):
    guard='guarded_v7'

    def test_unchanged_is_not_a_new_duration_and_cannot_dispute_old_value(self):
        m=memory();before=deepcopy(m.case_state['fact_ledger']['records'])
        bad=delta(fact('未变','口渴的时间没有变化',field='duration',about='t1:f1'))
        result=SemanticExtractor(FixedClient(bad,bad),recovery_mode=self.guard).run(m,'口渴的时间没有变化。')
        self.assertEqual(result['status'],'error')
        self.assertEqual(m.case_state['fact_ledger']['records'],before)

    def test_explicit_correction_cannot_assert_conflicting_attribute(self):
        m=memory();before=deepcopy(m.case_state['fact_ledger']['records'])
        text='口渴发生的时间我记错了，应该是三天。'
        bad=delta(fact('三天',text,field='duration',about='t1:f1'))
        result=SemanticExtractor(FixedClient(bad,bad),recovery_mode=self.guard).run(m,text)
        self.assertEqual(result['status'],'error')
        self.assertEqual(m.case_state['fact_ledger']['records'],before)

    def test_model_correction_replaces_old_attribute_without_program_filling(self):
        m=memory();text='口渴发生的时间我记错了，应该是三天。'
        bad=delta(fact('三天',text,field='duration',about='t1:f1'))
        correct=delta(fact('三天',text,field='duration',about='t1:f1',operation='correct',replaces='t1:f2'))
        result=SemanticExtractor(FixedClient(bad),repair_client=FixedClient(correct),recovery_mode=self.guard).run(m,text)
        self.assertEqual(result['status'],'ok');self.assertEqual(result['attempts'][0]['raw_delta'],bad)
        self.assertEqual(result['raw_delta'],correct)
        self.assertEqual(m.case_state['fact_view']['problems'][0]['attributes'][0]['value'],'三天')

    def test_same_old_value_and_new_unrelated_duration_are_not_rejected(self):
        for text,new in [('口渴的时间没有变化。',delta(fact('两天','口渴的时间没有变化',field='duration',about='t1:f1'))),
            ('肩痛三天。',delta(fact('肩痛'),fact('三天',id='f2',field='duration',about='f1')))]:
            with self.subTest(text=text):
                self.assertEqual(SemanticExtractor(FixedClient(new),recovery_mode=self.guard).run(memory(),text)['status'],'ok')

    def test_legacy_guard_is_reproducible(self):
        bad=delta(fact('未变','口渴的时间没有变化',field='duration',about='t1:f1'))
        self.assertEqual(SemanticExtractor(FixedClient(bad),recovery_mode='guarded_v6').run(memory(),'口渴的时间没有变化。')['status'],'ok')

    def test_explicit_uncertainty_is_not_treated_as_confirmed_new_attribute(self):
        new=delta(fact('不确定','程度不确定',field='severity',polarity='uncertain',about='t1:f1'))
        self.assertEqual(SemanticExtractor(FixedClient(new),recovery_mode=self.guard).run(memory(),'口渴的程度不确定。')['status'],'ok')


if __name__=='__main__':unittest.main()
