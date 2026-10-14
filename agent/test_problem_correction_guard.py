"""Explicit correction must use a replacement, never silently clear a risk."""
from copy import deepcopy
import unittest

from agent.semantic_extraction import SemanticExtractor
from agent.test_semantic_runtime import FixedClient, delta
from agent.graph import _run_risk_assessment
from llm.fact_contract_runtime import V2
from memory.memory import ConversationMemory
from memory.fact_ledger_runtime import statuses
from training.test_fact_contract import fact


class ProblemCorrectionTests(unittest.TestCase):
    def seed(self):
        memory=ConversationMemory();memory.enable_fact_mode(V2)
        inp=memory.fact_input('现在呼吸困难。')
        memory.apply_fact_delta(inp,delta(fact('呼吸困难')))
        return memory

    def actor(self,client):
        return SemanticExtractor(client,contract_version=V2,recovery_mode='guarded_v11',prompt_version='coverage_v5')

    def test_wrong_assert_is_retried_then_explicit_replacement_clears(self):
        m=self.seed()
        wrong=delta(fact('呼吸困难','刚才说错了，其实没有呼吸困难',polarity='negated'))
        right=delta(fact('呼吸困难','刚才说错了，其实没有呼吸困难',polarity='negated',operation='correct',replaces='t1:f1'))
        client=FixedClient(wrong,right)
        result=self.actor(client).run(m,'刚才说错了，其实没有呼吸困难。')
        self.assertEqual(result['status'],'ok')
        self.assertEqual(len(client.calls),2)
        self.assertEqual(result['attempts'][0]['raw_delta'],wrong)
        self.assertEqual(statuses(m.case_state['fact_ledger'])['t1:f1'],'superseded')
        self.assertEqual(_run_risk_assessment(m.case_state)['risk'],'UNKNOWN')

    def test_failed_repair_retains_old_positive_and_high(self):
        m=self.seed();before=deepcopy(m.case_state['fact_ledger']['records'])
        wrong=delta(fact('呼吸困难','刚才说错了，其实没有呼吸困难',polarity='negated'))
        result=self.actor(FixedClient(wrong,wrong)).run(m,'刚才说错了，其实没有呼吸困难。')
        self.assertEqual(result['status'],'error')
        self.assertEqual(m.case_state['fact_ledger']['records'],before)
        self.assertEqual(_run_risk_assessment(m.case_state)['risk'],'HIGH')

    def test_ordinary_negation_is_not_automatically_a_correction(self):
        m=self.seed();client=FixedClient(delta(fact('呼吸困难','没有呼吸困难',polarity='negated')))
        result=self.actor(client).run(m,'没有呼吸困难。')
        self.assertEqual(result['status'],'ok')
        self.assertEqual(len(client.calls),1)
        self.assertEqual(statuses(m.case_state['fact_ledger'])['t1:f1'],'disputed')

    def test_other_subject_correction_does_not_clear_patient(self):
        m=self.seed();client=FixedClient(delta(fact('呼吸困难','父亲刚才说错了，没有呼吸困难',polarity='negated',subject='other')))
        result=self.actor(client).run(m,'父亲刚才说错了，没有呼吸困难。')
        self.assertEqual(result['status'],'ok')
        self.assertEqual(len(client.calls),1)
        self.assertEqual(_run_risk_assessment(m.case_state)['risk'],'HIGH')

    def test_correcting_an_unrelated_attribute_does_not_force_symptom_replacement(self):
        m=self.seed();client=FixedClient(delta(fact('咳嗽','没有咳嗽',polarity='negated')))
        result=self.actor(client).run(m,'刚才年龄说错了，我也没有咳嗽。')
        self.assertEqual(result['status'],'ok')
        self.assertEqual(len(client.calls),1)
        self.assertEqual(_run_risk_assessment(m.case_state)['risk'],'HIGH')

    def test_new_occurrence_rejects_fabricated_correction_and_retries_assert(self):
        m=ConversationMemory();m.enable_fact_mode(V2)
        m.apply_fact_delta(m.fact_input('没有呼吸困难。'),delta(fact('呼吸困难','没有呼吸困难',polarity='negated')))
        wrong=delta(fact('呼吸困难','现在确实呼吸困难',operation='correct',replaces='t1:f1'))
        right=delta(fact('呼吸困难','现在确实呼吸困难'))
        result=self.actor(FixedClient(wrong,right)).run(m,'现在确实呼吸困难。')
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['attempts'][0]['error'],'explicit_new_occurrence_must_not_correct_prior_denial')
        self.assertEqual(statuses(m.case_state['fact_ledger'])['t1:f1'],'past')
        self.assertEqual(_run_risk_assessment(m.case_state)['risk'],'HIGH')

    def test_symptom_before_onset_rejects_fabricated_correction(self):
        m=ConversationMemory();m.enable_fact_mode(V2)
        m.apply_fact_delta(m.fact_input('没有腰痛。'),delta(fact('腰痛','没有腰痛',polarity='negated')))
        wrong=delta(fact('腰痛','腰痛是刚刚才开始的',operation='correct',replaces='t1:f1'))
        right=delta(fact('腰痛','腰痛是刚刚才开始的'))
        result=self.actor(FixedClient(wrong,right)).run(m,'腰痛是刚刚才开始的。')
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['attempts'][0]['error'],'explicit_new_occurrence_must_not_correct_prior_denial')
        self.assertEqual(statuses(m.case_state['fact_ledger'])['t1:f1'],'past')
        self.assertEqual(m.case_state['fact_ledger']['records'][1]['operation'],'assert')
        self.assertIsNone(m.case_state['fact_ledger']['records'][1]['replaces'])
