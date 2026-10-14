"""Identity across typed fields must not retain a stale respiratory denial."""
from copy import deepcopy
import unittest
from llm.fact_contract_runtime import V2
from memory.memory import ConversationMemory
from memory.fact_ledger_runtime import statuses,next_input
from training.test_fact_contract import fact
from agent.graph import _run_risk_assessment


class ConceptTemporalTests(unittest.TestCase):
    def advance(self,m,text,value,field,**changes):
        m.apply_fact_delta(m.fact_input(text),{'facts':[fact(value,text.strip('。'),field=field,**changes)]})

    def test_explicit_occurrence_ends_same_concept_denial_in_either_field(self):
        for a,b in [('symptoms','red_flags'),('red_flags','symptoms')]:
            with self.subTest(fields=(a,b)):
                m=ConversationMemory();m.enable_fact_mode(V2)
                self.advance(m,'没有呼吸困难。','呼吸困难',a,polarity='negated')
                old=deepcopy(m.case_state['fact_ledger']['records'][0])
                self.advance(m,'呼吸困难刚开始。','呼吸困难',b)
                ledger=m.case_state['fact_ledger']
                self.assertEqual(statuses(ledger),{'t1:f1':'past','t2:f1':'active'})
                self.assertEqual(ledger['records'][0],old)
                self.assertIsNone(ledger['records'][1]['replaces'])
                self.assertEqual(ledger['records'][1]['field'],b)
                self.assertEqual(_run_risk_assessment(m.case_state)['risk'],'HIGH')
                self.assertNotIn('t1:f1',[f['id'] for f in next_input(ledger,'继续')['known_facts']])

    def test_unmarked_opposites_are_conflict_not_new_occurrence(self):
        m=ConversationMemory();m.enable_fact_mode(V2)
        self.advance(m,'没有呼吸困难。','呼吸困难','symptoms',polarity='negated')
        self.advance(m,'呼吸困难。','呼吸困难','red_flags')
        self.assertEqual(set(statuses(m.case_state['fact_ledger']).values()),{'disputed'})

    def test_same_turn_opposites_cannot_end_denial_even_when_fields_differ(self):
        m=ConversationMemory();m.enable_fact_mode(V2)
        self.advance(m,'没有呼吸困难。','呼吸困难','symptoms',polarity='negated')
        m.apply_fact_delta(m.fact_input('呼吸困难刚开始，但没有呼吸困难。'),{'facts':[
            fact('呼吸困难','呼吸困难刚开始',field='red_flags'),
            fact('呼吸困难','没有呼吸困难',field='symptoms',polarity='negated',id='f2')]})
        self.assertNotEqual(statuses(m.case_state['fact_ledger'])['t1:f1'],'past')
        self.assertTrue(m.case_state['contradictions'])

    def test_scope_differences_and_related_but_nonidentical_concepts_do_not_merge(self):
        for value,field,changes in [('呼吸困难','red_flags',dict(subject='other')),
                ('呼吸困难','red_flags',dict(temporality='hypothetical')),
                ('持续胸痛','red_flags',{}),('便血','red_flags',{})]:
            old_value='胸痛' if value=='持续胸痛' else '黑便' if value=='便血' else '呼吸困难'
            m=ConversationMemory();m.enable_fact_mode(V2)
            self.advance(m,'没有'+old_value+'。',old_value,'symptoms',polarity='negated')
            self.advance(m,value+'刚开始。',value,field,**changes)
            self.assertEqual(statuses(m.case_state['fact_ledger'])['t1:f1'],'active')

    def test_existing_ledger_without_identity_opt_in_keeps_legacy_interpretation(self):
        m=ConversationMemory();m.enable_fact_mode(V2)
        m.case_state['fact_ledger'].pop('concept_identity_version',None)
        self.advance(m,'没有呼吸困难。','呼吸困难','symptoms',polarity='negated')
        self.advance(m,'呼吸困难刚开始。','呼吸困难','red_flags')
        self.assertEqual(set(statuses(m.case_state['fact_ledger']).values()),{'active'})
