import unittest
from copy import deepcopy
from agent.test_semantic_runtime import delta
from training.test_fact_contract import fact
from memory.memory import ConversationMemory
from memory.fact_ledger import risk_input
from memory.runtime_fact_projection import VERSION, derive_denied_red_flags


class RuntimeFactProjectionTests(unittest.TestCase):
    def test_negative_identity_projection_preserves_raw_ledger_input_and_risk(self):
        m = ConversationMemory(); m.enable_fact_mode()
        m.apply_fact_delta(m.fact_input('没有呼吸困难。'), delta(fact('呼吸困难', polarity='negated')))
        ledger = deepcopy(m.case_state['fact_ledger']); raw = deepcopy(m.case_state['fact_view'])
        inp = m.fact_input('鼻塞。'); risk = risk_input(ledger)
        m.enable_runtime_fact_projection(VERSION)
        self.assertEqual(m.case_state['fact_ledger'], ledger)
        self.assertEqual(m.fact_input('鼻塞。'), inp)
        self.assertEqual(risk_input(m.case_state['fact_ledger']), risk)
        self.assertEqual(m.case_state['raw_fact_view'], raw)
        derived = [f for f in m.case_state['fact_view']['facts'] if f['field'] == 'red_flags']
        self.assertEqual(len(derived), 1); self.assertEqual(derived[0]['polarity'], 'negated')
        self.assertEqual(derived[0]['derived_from'], 't1:f1')
        self.assertEqual(m.case_state['red_flags'], [])

    def test_is_idempotent_and_does_not_mutate_argument(self):
        view = dict(facts=[dict(fact('呼吸困难', polarity='negated'), derived_from=None)])
        saved = deepcopy(view); first, trace = derive_denied_red_flags(view)
        second, again = derive_denied_red_flags(first)
        self.assertEqual(view, saved); self.assertEqual(second, first)
        self.assertEqual(len(trace['added']), 1); self.assertEqual(again['added'], [])

    def test_finite_identity_scope_controls(self):
        controls = [dict(value='胸痛'), dict(value='黑便'), dict(polarity='present'),
                    dict(polarity='uncertain'), dict(subject='other'), dict(temporality='hypothetical'),
                    dict(temporality='historical'), dict(status='disputed'), dict(derived_from='old')]
        for changes in controls:
            with self.subTest(changes=changes):
                f = dict(fact('呼吸困难', polarity='negated'), derived_from=None); f.update(changes)
                view = dict(facts=[f]); projected, trace = derive_denied_red_flags(view)
                self.assertEqual(projected, view); self.assertEqual(trace['added'], [])

    def test_explicit_red_flag_negative_is_not_duplicated(self):
        symptom = dict(fact('呼吸困难', polarity='negated'), derived_from=None)
        red = dict(symptom, id='f2', field='red_flags')
        view = dict(facts=[symptom, red])
        self.assertEqual(derive_denied_red_flags(view)[0], view)

    def test_corrected_denial_does_not_leave_stale_derived_negative(self):
        m = ConversationMemory(); m.enable_fact_mode(); m.enable_runtime_fact_projection(VERSION)
        m.apply_fact_delta(m.fact_input('没有呼吸困难。'), delta(fact('呼吸困难', polarity='negated')))
        m.apply_fact_delta(m.fact_input('说错了，有呼吸困难。'), delta(fact('呼吸困难',
            operation='correct', replaces='t1:f1')))
        self.assertFalse(any(f['field'] == 'red_flags' and f['polarity'] == 'negated'
                             for f in m.case_state['fact_view']['facts']))
        self.assertEqual(m.case_state['runtime_fact_projection_trace']['added'], [])

    def test_requires_explicit_opt_in_and_supported_version(self):
        m = ConversationMemory()
        with self.assertRaises(ValueError): m.enable_runtime_fact_projection(VERSION)
        m.enable_fact_mode()
        with self.assertRaises(ValueError): m.enable_runtime_fact_projection('unknown')
        self.assertNotIn('raw_fact_view', m.case_state)


if __name__ == '__main__': unittest.main()
