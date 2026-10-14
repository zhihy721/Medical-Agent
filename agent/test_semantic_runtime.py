"""Real graph tests with fixed extraction outputs; no model-quality claims."""
from copy import deepcopy
import json
import unittest
from unittest.mock import Mock, patch

from agent.factory import create_agent
from agent.semantic_extraction import SemanticExtractor
from llm.clients import ModelResult
from memory.memory import ConversationMemory
from memory.session_store import InMemorySessionStore
from memory.fact_ledger import next_input, project, statuses
from training.test_fact_contract import advance, fact


class FixedClient:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []

    def call(self, messages, **kwargs):
        self.calls.append(deepcopy(messages))
        value = self.outputs.pop(0)
        if isinstance(value, ModelResult):
            return value
        return ModelResult(json.dumps(value, ensure_ascii=False), 'vllm', 'fixture-only', {}, 0, 'ok')


def delta(*facts):
    return {'facts': list(facts)}


class SemanticRuntimeTests(unittest.TestCase):
    def test_alias_audit_and_cross_turn_resolution(self):
        agent, _ = self.agent(delta(fact('头疼'), fact('三天', field='duration', id='f2', about='f1')),
            delta(fact('头痛')), delta(fact('头疼', '现在好了', operation='resolve',
                temporality='resolved', replaces='t1:f1')))
        agent.run('头疼三天。')
        agent.run('还是头痛。')
        ledger = agent.memory.get_case_state()['fact_ledger']
        self.assertEqual(ledger['turns'][0]['raw_delta']['facts'][0]['value'], '头疼')
        self.assertEqual(ledger['turns'][0]['normalized_delta']['facts'][0]['value'], '头痛')
        self.assertEqual(project(ledger)['problems'][0]['id'], 't1:f1')
        self.assertEqual(project(ledger)['problems'][0]['attributes'][0]['value'], '三天')
        agent.run('现在好了。')
        self.assertEqual(agent.get_graph_state().values['extraction_trace']['status'], 'ok')
        self.assertFalse(agent.memory.get_case_state()['fact_view']['problems'])

    def test_breathing_question_is_capped_and_preserves_safety(self):
        agent, _ = self.agent(delta(fact('呼吸困难', '喘不上气', field='red_flags')), delta(), delta())
        replies = [agent.run(text) for text in ('喘不上气。', '嗯。', '嗯。')]
        self.assertTrue(all('需要优先警惕' in reply for reply in replies))
        self.assertIn('若能够安全回复', replies[0])
        self.assertIn('若能够安全回复', replies[1])
        self.assertNotIn('若能够安全回复', replies[2])
        state = agent.memory.get_case_state()
        self.assertEqual(state['interpretation_question_counts'], {'t1:f1': 2})
        self.assertEqual(state['risk_level'], 'HIGH')
        self.assertEqual(state['fact_ledger']['turns'][0]['interpretations'][0]['fact_id'], 't1:f1')

    def test_colloquial_candidate_correction_and_resolution_are_distinct(self):
        for operation, polarity, time, evidence, expected in (
                ('correct', 'negated', 'current', '说错了，没有气短', 'UNKNOWN'),
                ('resolve', 'present', 'resolved', '现在缓解了', 'HIGH')):
            with self.subTest(operation=operation):
                agent, _ = self.agent(delta(fact('气短', '喘不过气')),
                    delta(fact('气短', evidence, operation=operation, polarity=polarity,
                               temporality=time, replaces='t1:f1')))
                agent.run('喘不过气。')
                self.assertEqual(agent.memory.get_case_state()['risk_level'], 'HIGH')
                response = agent.run(evidence + '。')
                self.assertEqual(agent.memory.get_case_state()['risk_level'], expected)
                self.assertNotIn('若能够安全回复', response)

    def setUp(self):
        self.guideline = patch('agent.graph._run_guideline', return_value={'advice': []})
        self.hospital = patch('agent.runtime_utils._nearby_hospital_lines', return_value=[])
        self.guideline.start(); self.hospital.start()
        self.addCleanup(self.guideline.stop); self.addCleanup(self.hospital.stop)

    def agent(self, *outputs, memory=None, responder=None):
        client, legacy = FixedClient(*outputs), Mock()
        legacy.last_provider_used = 'deepseek'
        legacy.extract_json.side_effect = AssertionError('legacy extraction must not run')
        legacy.call.side_effect = AssertionError('extractor provider must not select responder')
        agent = create_agent(legacy, memory or ConversationMemory(), fact_extractor=SemanticExtractor(client),
                             responder=responder, response_mode='model' if responder else 'template')
        return agent, client

    def test_two_turn_real_graph_corrects_only_the_referenced_problem(self):
        agent, client = self.agent(
            delta(fact('咳嗽'), fact('三天', field='duration', id='f2', about='f1'),
                  fact('头痛', id='f3'), fact('一天', field='duration', id='f4', about='f3')),
            delta(fact('头痛'), fact('两天', '其实两天', field='duration', id='f2',
                                   about='f1', operation='correct', replaces='t1:f4')))
        agent.run('咳嗽三天，头痛一天。')
        before = agent.memory.get_case_state()['fact_ledger']['records']
        agent.run('头痛其实两天，之前说错了。')
        state = agent.memory.get_case_state()
        values = {p['value']: [a['value'] for a in p['attributes']] for p in state['fact_view']['problems']}
        self.assertEqual(values, {'咳嗽': ['三天'], '头痛': ['两天']})
        self.assertEqual(state['fact_ledger']['records'][:4], before)
        self.assertEqual(state['chief_complaint'], '')
        self.assertEqual(state['chief_complaint_candidates'], ['咳嗽', '头痛'])
        self.assertNotIn('chief_complaint', state['missing_slots'])
        self.assertEqual(agent.get_graph_state().values['extraction_trace']['status'], 'ok')
        self.assertEqual(len(client.calls), 2)

    def test_repeated_problem_has_stable_id_and_correction_clears_all_copies(self):
        agent, _ = self.agent(delta(fact('呼吸困难', '无法呼吸', field='red_flags')),
            delta(fact('呼吸困难', '无法呼吸', field='red_flags')),
            delta(fact('呼吸困难', '两次都说错了，没有呼吸困难', field='red_flags', polarity='negated',
                       operation='correct', replaces='t1:f1'), fact('咳嗽', id='f2')))
        agent.run('无法呼吸。'); agent.run('还是无法呼吸。')
        state = agent.memory.get_case_state()
        self.assertEqual(len(state['fact_view']['problems']), 1)
        self.assertEqual(state['fact_view']['problems'][0]['id'], 't1:f1')
        response = agent.run('两次都说错了，没有呼吸困难，只有咳嗽。')
        state = agent.memory.get_case_state()
        self.assertEqual(state['risk_level'], 'LOW')
        self.assertEqual(statuses(state['fact_ledger'])['t2:f1'], 'superseded')
        self.assertNotIn('呼吸困难', state['red_flags'])
        self.assertNotIn('需要优先警惕', response)

    def test_resolved_red_flag_remains_high_through_actual_graph(self):
        agent, _ = self.agent(delta(fact('呼吸困难', '无法呼吸', field='red_flags')),
            delta(fact('呼吸困难', '刚才确实无法呼吸，现在缓解了', field='red_flags',
                       operation='resolve', replaces='t1:f1', temporality='resolved')))
        agent.run('无法呼吸。')
        response = agent.run('刚才确实无法呼吸，现在缓解了。')
        self.assertEqual(agent.memory.get_case_state()['risk_level'], 'HIGH')
        self.assertIn('已缓解', response)
        self.assertEqual(agent.get_graph_state().values['action_result']['name'], 'risk_escalation')

    def test_invalid_output_is_visible_and_does_not_mutate_old_facts(self):
        agent, _ = self.agent(delta(fact('咳嗽')), delta(fact('呼吸困难', '不存在的证据', field='red_flags')),
                             delta(fact('三天', field='duration', about='t1:f1')))
        agent.run('咳嗽。')
        before = agent.memory.get_case_state()['fact_ledger']['records']
        response = agent.run('继续。')
        self.assertIn('未能可靠识别', response)
        self.assertEqual(agent.memory.get_case_state()['fact_ledger']['records'], before)
        rejected = agent.memory.get_case_state()['fact_ledger']['turns'][-1]
        self.assertIn('不存在的证据', rejected['model_result']['content'])
        agent.run('三天。')
        state = agent.memory.get_case_state()
        self.assertEqual(state['duration'], '三天')
        self.assertEqual(state['fact_ledger']['records'][-1]['source_turn'], 3)
        self.assertEqual(state['fact_extraction_error'], '')

    def test_failed_provider_does_not_remove_known_high_risk(self):
        agent, _ = self.agent(delta(fact('呼吸困难', '无法呼吸', field='red_flags')),
                             ModelResult('', 'vllm', 'fixture', {}, 0, 'timeout'))
        agent.run('无法呼吸。')
        response = agent.run('现在呢？')
        self.assertIn('就医', response)
        self.assertEqual(agent.memory.get_case_state()['risk_level'], 'HIGH')
        self.assertEqual(agent.get_graph_state().values['extraction_trace']['status'], 'error')

    def test_targeted_correction_resolves_previous_dispute(self):
        agent, _ = self.agent(delta(fact('咳嗽'), fact('三天', field='duration', id='f2', about='f1')),
            delta(fact('两天', field='duration', about='t1:f1')),
            delta(fact('两天', '三天说错了，实际两天', field='duration', about='t1:f1',
                       operation='correct', replaces='t1:f2')))
        agent.run('咳嗽三天。'); agent.run('两天。')
        self.assertTrue(agent.memory.get_case_state()['contradictions'])
        agent.run('三天说错了，实际两天。')
        self.assertEqual(agent.memory.get_case_state()['duration'], '两天')
        self.assertFalse(agent.memory.get_case_state()['contradictions'])

    def test_explicit_binding_resolves_previously_unbound_duration(self):
        agent, _ = self.agent(delta(fact('头痛'), fact('咳嗽', id='f2'), fact('三天', field='duration', id='f3')),
            delta(fact('三天', '三天指的是咳嗽', field='duration', about='t1:f2',
                       operation='correct', replaces='t1:f3')))
        agent.run('头痛又咳嗽，三天。')
        self.assertEqual(agent.get_graph_state().values['action_result']['name'], 'clarify_conflict')
        agent.run('三天指的是咳嗽。')
        self.assertFalse(agent.memory.get_case_state()['fact_view']['unbound_attributes'])
        self.assertFalse(agent.memory.get_case_state()['contradictions'])

    def test_profile_provenance_form_edit_and_session_restore(self):
        store = InMemorySessionStore()
        memory = ConversationMemory(session_store=store, session_id='semantic-restore')
        memory.apply_manual_profile({'age': '70', 'past_history': ['高血压']})
        agent, client = self.agent(delta(fact('咳嗽')), memory=memory)
        agent.run('咳嗽。')
        inp = json.loads(client.calls[0][1]['content'])
        self.assertEqual({f['source'] for f in inp['known_facts']}, {'profile_prefill'})
        memory.apply_manual_profile({'age': '30', 'past_history': []})
        restored = ConversationMemory(session_store=store, session_id='semantic-restore',
                                      profile_store=memory.profile_store, user_id=memory.user_id)
        second, _ = self.agent(delta(), memory=restored)
        second.run('继续。')
        self.assertEqual(restored.get_case_state()['age'], '30')
        self.assertEqual(restored.get_case_state()['past_history'], [])
        self.assertEqual(restored.get_case_state()['risk_level'], 'LOW')
        self.assertTrue(any(f['source'] == 'manual_profile' for f in restored.fact_input('好的')['known_facts']))

    def test_old_session_cannot_silently_switch_or_semantic_session_fall_back(self):
        memory = ConversationMemory(); memory.add_user('旧问诊。')
        with self.assertRaisesRegex(ValueError, 'new session'):
            self.agent(delta(), memory=memory)
        agent, _ = self.agent(delta())
        with self.assertRaisesRegex(ValueError, 'requires semantic'):
            create_agent(Mock(), agent.memory)

    def test_independent_responder_does_not_use_extractor_provider_switch(self):
        responder = FixedClient(ModelResult('独立回复：咳嗽多久了？', 'vllm', 'responder', {}, 0, 'ok'))
        agent, extractor = self.agent(delta(fact('咳嗽')), responder=responder)
        self.assertEqual(agent.run('咳嗽。'), '独立回复：咳嗽多久了？')
        self.assertEqual(len(responder.calls), 1)
        self.assertEqual(len(extractor.calls), 1)

    def test_high_never_calls_even_explicit_responder(self):
        responder = FixedClient()
        agent, _ = self.agent(delta(fact('呼吸困难', '无法呼吸', field='red_flags')), responder=responder)
        agent.run('无法呼吸。')
        self.assertFalse(responder.calls)

    def test_repeated_facts_compact_context_but_preserve_evidence_history(self):
        from memory.fact_ledger import empty_ledger
        ledger = empty_ledger()
        for _ in range(35):
            ledger = advance(ledger, '咳嗽。', fact('咳嗽'))
        self.assertEqual(len(ledger['records']), 35)
        self.assertEqual(len(next_input(ledger, '继续。')['known_facts']), 1)
        self.assertEqual(len(project(ledger)['problems']), 1)

    def test_reaffirmed_problem_does_not_discard_unmentioned_attributes(self):
        from memory.fact_ledger import empty_ledger
        ledger = advance(empty_ledger(), '咳嗽中度，三天。', fact('咳嗽'),
            fact('中度', field='severity', id='f2', about='f1'), fact('三天', field='duration', id='f3', about='f1'))
        ledger = advance(ledger, '咳嗽其实两天。',
            fact('咳嗽', operation='correct', replaces='t1:f1'),
            fact('两天', field='duration', id='f2', about='f1', operation='correct', replaces='t1:f3'))
        problem = project(ledger)['problems'][0]
        self.assertEqual(problem['id'], 't1:f1')
        self.assertEqual({f['value'] for f in problem['attributes']}, {'中度', '两天'})

    def test_pulse_input_keeps_semantic_facts_and_survives_next_turn(self):
        agent, _ = self.agent(delta(fact('咳嗽')), delta())
        agent.run('咳嗽。')
        before = agent.memory.get_case_state()['fact_ledger']['records']
        agent.memory.update_pulse_data({'pulse_summary': '测试脉象', 'pulse_source': 'fixture'})
        agent.run('继续。')
        self.assertEqual(agent.memory.get_case_state()['pulse_summary'], '测试脉象')
        self.assertEqual(agent.memory.get_case_state()['fact_ledger']['records'], before)

    def test_restoration_does_not_refill_a_corrected_profile_fact(self):
        store = InMemorySessionStore()
        memory = ConversationMemory(session_store=store, session_id='profile-correction')
        memory.apply_manual_profile({'past_history': ['高血压']})
        agent, _ = self.agent(delta(fact('高血压', '之前填错了，没有高血压', field='past_history',
                                        polarity='negated', operation='correct', replaces='profile:1')), memory=memory)
        agent.run('之前填错了，没有高血压。')
        restored = ConversationMemory(session_store=store, session_id='profile-correction',
                                      profile_store=memory.profile_store, user_id=memory.user_id)
        self.assertEqual(restored.get_case_state()['past_history'], [])


if __name__ == '__main__':
    unittest.main()
