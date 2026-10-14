"""New occurrence, correction and resolution stay distinct in immutable history."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from agent.graph import _run_risk_assessment
from llm.fact_contract_runtime import V2
from memory.memory import ConversationMemory
from memory.fact_ledger_runtime import statuses, next_input
from memory.rollout_store import RolloutSessionStore
from training.test_fact_contract import fact


class TemporalTests(unittest.TestCase):
    def advance(self,m,text,*facts):
        m.apply_fact_delta(m.fact_input(text),{'facts':list(facts)})

    def memory(self):
        m=ConversationMemory();m.enable_fact_mode(V2);return m

    def test_new_occurrence_preserves_old_denial_without_dispute(self):
        m=self.memory();self.advance(m,'我没有呼吸困难。',fact('呼吸困难','没有呼吸困难',polarity='negated'))
        self.advance(m,'现在咳嗽。',fact('咳嗽','现在咳嗽'))
        before=deepcopy(m.case_state['fact_ledger']['records'][0])
        self.advance(m,'现在确实呼吸困难。',fact('呼吸困难','现在确实呼吸困难'))
        ledger=m.case_state['fact_ledger'];states=statuses(ledger)
        self.assertEqual(ledger['records'][0],before)
        self.assertEqual(states['t1:f1'],'past')
        self.assertEqual(states['t3:f1'],'active')
        self.assertEqual(m.case_state['contradictions'],[])
        self.assertEqual(_run_risk_assessment(m.case_state)['risk'],'HIGH')
        self.assertNotIn('t1:f1',[f['id'] for f in next_input(ledger,'下一条')['known_facts']])
        self.assertEqual(m.case_state['fact_view']['previous_current_facts'][0]['source_turn'],1)

    def test_resolution_keeps_episode_high_without_erasing_prior_denial(self):
        m=self.memory();self.advance(m,'没有呼吸困难。',fact('呼吸困难','没有呼吸困难',polarity='negated'))
        self.advance(m,'现在有呼吸困难。',fact('呼吸困难','现在有呼吸困难'))
        self.advance(m,'刚才呼吸困难已缓解。',fact('呼吸困难','刚才呼吸困难已缓解',temporality='resolved',operation='resolve',replaces='t2:f1'))
        self.assertEqual(statuses(m.case_state['fact_ledger'])['t1:f1'],'past')
        self.assertEqual(_run_risk_assessment(m.case_state)['risk'],'HIGH')

    def test_ambiguous_or_same_turn_opposites_remain_disputed(self):
        for text in ['呼吸困难。','刚才说错了，现在呼吸困难。','不确定现在是不是呼吸困难。']:
            m=self.memory();self.advance(m,'没有呼吸困难。',fact('呼吸困难','没有呼吸困难',polarity='negated'))
            self.advance(m,text,fact('呼吸困难',text.strip('。')))
            self.assertEqual(statuses(m.case_state['fact_ledger'])['t1:f1'],'disputed')
        m=self.memory();self.advance(m,'没有呼吸困难。',fact('呼吸困难','没有呼吸困难',polarity='negated'))
        self.advance(m,'现在有呼吸困难，但没有呼吸困难。',fact('呼吸困难','现在有呼吸困难'),
            fact('呼吸困难','没有呼吸困难',polarity='negated',id='f2'))
        self.assertTrue(m.case_state['contradictions'])
        self.assertNotEqual(statuses(m.case_state['fact_ledger'])['t1:f1'],'past')

    def test_other_hypothetical_and_quoted_do_not_end_denial(self):
        for text,changes in [('父亲现在呼吸困难。',dict(subject='other')),
                             ('如果现在呼吸困难。',dict(temporality='hypothetical')),
                             ('引用“现在呼吸困难”。',{})]:
            m=self.memory();self.advance(m,'没有呼吸困难。',fact('呼吸困难','没有呼吸困难',polarity='negated'))
            self.advance(m,text,fact('呼吸困难',text.strip('。'),**changes))
            self.assertNotEqual(statuses(m.case_state['fact_ledger'])['t1:f1'],'past')

    def test_explicit_correct_is_superseded_not_past_and_old_sessions_keep_policy(self):
        m=self.memory();self.advance(m,'没有呼吸困难。',fact('呼吸困难','没有呼吸困难',polarity='negated'))
        self.advance(m,'刚才说错了，确实呼吸困难。',fact('呼吸困难','刚才说错了，确实呼吸困难',operation='correct',replaces='t1:f1'))
        self.assertEqual(statuses(m.case_state['fact_ledger'])['t1:f1'],'superseded')
        m=self.memory();m.case_state['fact_ledger'].pop('temporal_policy_version',None)
        self.advance(m,'没有呼吸困难。',fact('呼吸困难','没有呼吸困难',polarity='negated'))
        self.advance(m,'现在呼吸困难。',fact('呼吸困难','现在呼吸困难'))
        self.assertEqual(statuses(m.case_state['fact_ledger'])['t1:f1'],'disputed')

    def test_narrow_evidence_cannot_hide_source_uncertainty_or_correction(self):
        for text in ['刚才说错了，我现在呼吸困难。','不确定，现在呼吸困难是否属实。','引用“现在呼吸困难”。']:
            m=self.memory();self.advance(m,'没有呼吸困难。',fact('呼吸困难','没有呼吸困难',polarity='negated'))
            self.advance(m,text,fact('呼吸困难','现在呼吸困难'))
            self.assertNotEqual(statuses(m.case_state['fact_ledger'])['t1:f1'],'past')

    def test_omitted_same_turn_denial_still_prevents_temporal_clearance(self):
        m=self.memory();self.advance(m,'没有呼吸困难。',fact('呼吸困难','没有呼吸困难',polarity='negated'))
        self.advance(m,'现在呼吸困难，但是没有呼吸困难。',fact('呼吸困难','现在呼吸困难'))
        self.assertEqual(statuses(m.case_state['fact_ledger'])['t1:f1'],'disputed')

    def test_temporal_history_survives_full_rollout_index_rebuild(self):
        with tempfile.TemporaryDirectory() as root:
            store=RolloutSessionStore(root)
            m=ConversationMemory(session_store=store,session_id='s',user_id='u');m.enable_fact_mode(V2)
            for text,p in [('没有呼吸困难。','negated'),('现在确实呼吸困难。','present')]:
                m.add_user(text);self.advance(m,text,fact('呼吸困难',text.strip('。'),polarity=p))
                m.add_assistant('已记录。')
            before=deepcopy(m.case_state)
            store.database.unlink()
            restored=ConversationMemory(session_store=RolloutSessionStore(root),session_id='s',user_id='u')
            self.assertEqual(restored.case_state,before)
            self.assertEqual(statuses(restored.case_state['fact_ledger'])['t1:f1'],'past')
            self.assertEqual(len(restored.get_archived_messages()),4)
