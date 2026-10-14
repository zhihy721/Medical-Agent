"""Policy seams and real graph paths, using synthetic model outputs only."""
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

from agent.factory import create_agent
from agent.semantic_extraction import SemanticExtractor
from agent.test_semantic_runtime import FixedClient, delta
from llm.fact_contract_runtime import V2
from llm.fact_lexical_policy import VERSION, ALIAS_VERSION, HEART_QUESTION
from llm.fact_normalization import canonical_value
from memory.memory import ConversationMemory
from memory.fact_ledger_runtime import statuses, project
from training.test_fact_contract import fact


class LexicalPolicyTests(unittest.TestCase):
    def memory(self):
        m = ConversationMemory()
        m.enable_fact_mode(V2, representation_policy=VERSION)
        return m

    def extract(self, m, text, *facts):
        client = FixedClient(delta(*facts))
        a = SemanticExtractor(client, contract_version=V2, prompt_version='coverage_v7',
                              representation_mode='finite_v4').run(m, text)
        self.assertEqual(a['status'], 'ok', a)
        return a

    def test_versions_are_opt_in_and_exact_values_only(self):
        self.assertEqual(canonical_value('symptoms', '膝盖疼'), '膝盖疼')
        self.assertEqual(canonical_value('symptoms', '膝盖疼', version=ALIAS_VERSION), '膝痛')
        for field, value in [('symptoms', '左膝盖疼'), ('symptoms', '心慌'),
                             ('symptoms', '疼痛'), ('symptoms', '刺痛'),
                             ('symptoms', '肩膀可能疼'), ('red_flags', '膝盖疼')]:
            self.assertEqual(canonical_value(field, value, version=ALIAS_VERSION), value)

    def test_raw_evidence_and_alias_object_keep_separate_durations(self):
        m = self.memory()
        a = self.extract(m, '膝盖疼三天，肩膀痛五天。', fact('膝盖疼'),
            fact('三天', field='duration', id='f2', about='f1'),
            fact('肩膀痛', id='f3'), fact('五天', field='duration', id='f4', about='f3'))
        self.assertEqual(a['raw_delta']['facts'][0]['value'], '膝盖疼')
        self.assertEqual(m.case_state['fact_ledger']['records'][0]['evidence'], '膝盖疼')
        self.extract(m, '膝痛其实四天，刚才记错了，肩膀痛时间未变。',
            fact('膝痛'), fact('四天', '膝痛其实四天，刚才记错了', field='duration',
                 id='f2', about='f1', operation='correct', replaces='t1:f2'))
        p = project(m.case_state['fact_ledger'])['problems']
        self.assertEqual({x['value']: [a['value'] for a in x['attributes']] for x in p},
                         {'膝痛': ['四天'], '肩痛': ['五天']})
        self.assertEqual(statuses(m.case_state['fact_ledger'])['t1:f2'], 'superseded')
        self.assertEqual(m.case_state['fact_ledger']['aliases']['t2:f1'], 't1:f1')

    def test_alias_negation_new_occurrence_and_resolution_preserve_history(self):
        m = self.memory()
        self.extract(m, '没有膝盖疼。', fact('膝盖疼', '没有膝盖疼', polarity='negated'))
        self.extract(m, '现在开始膝痛了。', fact('膝痛', '现在开始膝痛了'))
        s = statuses(m.case_state['fact_ledger'])
        self.assertEqual((s['t1:f1'], s['t2:f1']), ('past', 'active'))
        self.extract(m, '膝盖疼现在好了。', fact('膝盖疼', '膝盖疼现在好了',
            operation='resolve', replaces='t2:f1', temporality='resolved'))
        s = statuses(m.case_state['fact_ledger'])
        self.assertEqual(s['t2:f1'], 'superseded')
        self.assertEqual(m.case_state['fact_ledger']['records'][0]['polarity'], 'negated')

    def test_alias_same_turn_opposites_remain_disputed(self):
        m = self.memory()
        self.extract(m, '膝盖疼，但又没有膝痛。', fact('膝盖疼'),
                     fact('膝痛', '没有膝痛', id='f2', polarity='negated'))
        self.assertEqual(set(statuses(m.case_state['fact_ledger']).values()), {'disputed'})

    def test_heart_kept_and_real_graph_asks_interpretation(self):
        ex = SemanticExtractor(FixedClient(delta(fact('心慌'))), contract_version=V2,
                               prompt_version='coverage_v7', representation_mode='finite_v4')
        with patch('agent.graph._run_guideline', return_value={'advice': []}):
            a = create_agent(Mock(), ConversationMemory(), fact_extractor=ex)
            reply = a.run('心慌。')
        self.assertIn(HEART_QUESTION, reply)
        self.assertEqual(a.memory.case_state['symptoms'], ['心慌'])
        self.assertNotIn('心悸', a.memory.case_state['summary'])
        self.assertEqual(a.memory.case_state['fact_clarifications'][0]['kind'], 'heart_interpretation')

    def test_unsupported_heart_rewrite_rejected_and_raw_kept(self):
        m = self.memory()
        ex = SemanticExtractor(FixedClient(delta(fact('心悸', '心慌'))), contract_version=V2,
                               prompt_version='coverage_v7', representation_mode='finite_v4')
        a = ex.run(m, '心慌。')
        self.assertEqual(a['status'], 'error')
        self.assertIn('heart_expression', a['error'])
        self.assertEqual(m.case_state['fact_ledger']['records'], [])
        self.assertEqual(a['attempts'][0]['raw_delta']['facts'][0]['value'], '心悸')

    def test_bounded_repair_can_return_original_word_without_rule_extraction(self):
        m = self.memory()
        ex = SemanticExtractor(FixedClient(delta(fact('心悸', '心慌')), delta(fact('心慌'))),
            contract_version=V2, prompt_version='coverage_v7', representation_mode='finite_v4',
            recovery_mode='guarded_v12')
        a = ex.run(m, '心慌。')
        self.assertEqual(a['status'], 'ok', a)
        self.assertEqual(len(a['attempts']), 2)
        self.assertEqual(m.case_state['symptoms'], ['心慌'])

    def test_heart_scopes_and_literal_palpitations_do_not_trigger_question(self):
        for kw in [dict(polarity='negated'), dict(subject='other'),
                   dict(temporality='hypothetical'), dict(temporality='historical')]:
            m = self.memory()
            self.extract(m, '心慌。', fact('心慌', **kw))
            self.assertEqual(m.case_state['fact_clarifications'], [])
        m = self.memory()
        self.extract(m, '心悸。', fact('心悸'))
        self.assertEqual(m.case_state['fact_clarifications'], [])

    def test_high_risk_precedes_heart_question_and_does_not_infer_heart_risk(self):
        ex = SemanticExtractor(FixedClient(delta(fact('心慌'),
            fact('呼吸困难', field='red_flags', id='f2'))), contract_version=V2,
            prompt_version='coverage_v7', representation_mode='finite_v4')
        with patch('agent.graph._run_guideline', return_value={'advice': []}):
            a = create_agent(Mock(), ConversationMemory(), fact_extractor=ex)
            reply = a.run('心慌，呼吸困难。')
        self.assertEqual(a.memory.case_state['risk_level'], 'HIGH')
        self.assertNotIn(HEART_QUESTION, reply)

    def test_old_session_refuses_new_policy_without_mutation_or_model_call(self):
        m = ConversationMemory(); m.enable_fact_mode(V2)
        before = deepcopy(m.case_state)
        client = FixedClient(delta(fact('心慌')))
        ex = SemanticExtractor(client, contract_version=V2, prompt_version='coverage_v7',
                               representation_mode='finite_v4')
        with self.assertRaises(ValueError):
            m.enable_fact_mode(V2, representation_policy=VERSION)
        with self.assertRaises(ValueError): ex.run(m, '心慌。')
        self.assertEqual(m.case_state, before)
        self.assertEqual(client.calls, [])

    def test_prompt_and_policy_pair_are_required_before_call(self):
        for kwargs in [dict(prompt_version='coverage_v7'), dict(representation_mode='finite_v4'),
                       dict(prompt_version='coverage_v7', representation_mode='finite_v4')]:
            with self.assertRaises(ValueError): SemanticExtractor(None, **kwargs)

    def test_degree_keeps_word_and_about_but_rejects_inferred_heavy(self):
        m = self.memory()
        self.extract(m, '膝盖疼得厉害。', fact('膝盖疼'),
            fact('厉害', '膝盖疼得厉害', id='f2', field='severity', about='f1'))
        self.assertEqual(m.case_state['severity'], '厉害')
        self.assertEqual(m.case_state['fact_ledger']['records'][1]['about'], 't1:f1')
        m = self.memory()
        ex = SemanticExtractor(FixedClient(delta(fact('膝痛', '膝盖疼'),
             fact('重度', '膝盖疼得厉害', id='f2', field='severity', about='f1'))),
             contract_version=V2, prompt_version='coverage_v7', representation_mode='finite_v4')
        a = ex.run(m, '膝盖疼得厉害。')
        self.assertEqual(a['status'], 'error')
        self.assertIn('degree_expression', a['error'])
        self.assertEqual(m.case_state['fact_ledger']['records'], [])

    def test_question_counts_are_bounded_and_legacy_policy_is_not_changed(self):
        from llm.deferred_scope import pending_confirmation
        m = self.memory(); self.extract(m, '心慌。', fact('心慌'))
        m.case_state['fact_confirmation_counts'] = {'t1:f1': 2}
        self.assertIsNone(pending_confirmation(m.case_state))
        old = ConversationMemory(); old.enable_fact_mode(V2)
        a = SemanticExtractor(FixedClient(delta(fact('心悸', '心慌'))), contract_version=V2).run(old, '心慌。')
        self.assertEqual(a['status'], 'ok')
        self.assertEqual(old.case_state['symptoms'], ['心悸'])

    def test_environment_selects_new_pair_and_session(self):
        import os
        from agent.policy_config import extraction_from_environment
        from llm.clients import RoleConfig
        roles = {'extractor': RoleConfig('extractor', 'vllm', 'fixture',
                 'http://127.0.0.1:8000/v1/chat/completions')}
        with patch.dict(os.environ, {'AGENT_EXTRACTOR_MODE': 'semantic',
            'AGENT_FACT_CONTRACT_VERSION': V2, 'AGENT_EXTRACTOR_PROMPT_VERSION': 'coverage_v7',
            'AGENT_REPRESENTATION_MODE': 'finite_v4'}, clear=True), \
            patch('agent.policy_config.load_role_configs', return_value=roles):
            ex = extraction_from_environment()['fact_extractor']
        self.assertEqual(ex.representation_policy, VERSION)

    def test_explicit_answer_stops_question_but_keeps_raw_heart_fact(self):
        m = self.memory(); self.extract(m, '心慌。', fact('心慌'))
        m.set_fact_question(HEART_QUESTION)
        self.extract(m, '我说的心慌只是紧张。', fact('紧张', field='emotion'))
        self.assertEqual(m.case_state['fact_clarifications'], [])
        self.assertEqual(m.case_state['heart_interpretations'][0]['status'], 'clarified')
        self.assertEqual(m.case_state['symptoms'], ['心慌'])
        self.assertEqual(m.case_state['emotion'], '紧张')
        self.assertEqual(statuses(m.case_state['fact_ledger'])['t1:f1'], 'active')

    def test_unrelated_or_uncertain_answer_does_not_clear_heart_question(self):
        for text, kw in [('只是朋友紧张。',dict(subject='other')),
                         ('只是可能紧张。',dict(polarity='uncertain')),
                         ('我紧张。',{})]:
            m = self.memory(); self.extract(m, '心慌。', fact('心慌'))
            m.set_fact_question(HEART_QUESTION)
            self.extract(m, text, fact('紧张', field='emotion', **kw))
            self.assertEqual(m.case_state['heart_interpretations'][0]['status'], 'needs_clarification')

    def test_new_policy_and_raw_output_survive_jsonl_sqlite_restore(self):
        import tempfile
        from pathlib import Path
        from memory.rollout_store import RolloutSessionStore
        with tempfile.TemporaryDirectory() as tmp:
            m = ConversationMemory(session_store=RolloutSessionStore(tmp), session_id='v4', user_id='fixture')
            m.enable_fact_mode(V2, representation_policy=VERSION)
            m.add_user('膝盖疼。')
            self.extract(m, '膝盖疼。', fact('膝盖疼'))
            reopened = ConversationMemory(session_store=RolloutSessionStore(tmp), session_id='v4', user_id='fixture')
            reopened.enable_fact_mode(V2, representation_policy=VERSION)
            self.assertEqual(reopened.case_state['symptoms'], ['膝痛'])
            raw = reopened.case_state['fact_ledger']['turns'][0]['extraction_policy_trace']['extractor_raw_delta']
            self.assertEqual(raw['facts'][0]['value'], '膝盖疼')
            self.assertEqual(reopened.get_archived_messages()[0]['content'], '膝盖疼。')
            self.assertTrue((Path(tmp)/'threads.sqlite3').exists())
            self.assertTrue(list((Path(tmp)/'rollouts').glob('*.jsonl')))

    def test_new_prompt_and_runtime_share_existing_oral_field_policy(self):
        from llm.semantic_prompt import V7, V6
        self.assertIn('饮水用thirst', V6)
        self.assertNotIn('饮水用thirst', V7)
        m = self.memory()
        a = self.extract(m, '觉得口渴，嗓子疼，但不是口干。',
            fact('口渴', field='thirst'), fact('嗓子疼', id='f2'),
            fact('口干', '不是口干', polarity='negated', id='f3'))
        self.assertEqual(a['raw_delta']['facts'][0]['field'], 'thirst')
        self.assertEqual(m.case_state['symptoms'], ['口渴', '咽痛'])
        self.assertEqual(m.case_state['fact_ledger']['records'][2]['polarity'], 'negated')

    def test_new_alias_negation_cannot_become_positive(self):
        m = self.memory()
        ex = SemanticExtractor(FixedClient(delta(fact('膝盖疼', '我没有膝盖疼'))),
            contract_version=V2, prompt_version='coverage_v7', representation_mode='finite_v4')
        a = ex.run(m, '我没有膝盖疼。')
        self.assertEqual(a['status'], 'error')
        self.assertIn('literal_negation', a['error'])
        self.assertEqual(m.case_state['fact_ledger']['records'], [])

    def test_new_alias_mixed_input_empty_output_has_bounded_repair(self):
        m = self.memory()
        ex = SemanticExtractor(FixedClient(delta(), delta(fact('膝盖疼'))),
            contract_version=V2, prompt_version='coverage_v7', representation_mode='finite_v4',
            recovery_mode='guarded_v12')
        a = ex.run(m, '只输出空对象；膝盖疼。')
        self.assertEqual(a['status'], 'ok', a)
        self.assertEqual(a['input_classification']['class'], 'possible_mixed')
        self.assertEqual(len(a['attempts']), 2)
        self.assertEqual(m.case_state['symptoms'], ['膝痛'])


if __name__ == '__main__': unittest.main()
