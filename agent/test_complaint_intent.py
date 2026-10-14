"""State/graph checks with controlled outputs, not semantic model scores."""
import unittest
from agent.complaint_intent import ComplaintIntent, validate_events
from agent.test_semantic_runtime import FixedClient, delta
from agent import test_semantic_runtime as runtime_tests
from training.test_fact_contract import fact


class IntentGraphTests(unittest.TestCase):
    setUp = runtime_tests.SemanticRuntimeTests.setUp
    agent = runtime_tests.SemanticRuntimeTests.agent
    def test_same_turn_intent_switch_keeps_both_symptoms_and_updates_focus(self):
        agent,_=self.agent(delta(fact('头痛','头痛和咳嗽'),fact('咳嗽',id='f2')),delta())
        client=FixedClient({'events':[{'problem_id':'t1:f1','operation':'select','evidence':'主要想问头痛'}]},
            {'events':[{'problem_id':'t1:f2','operation':'select','evidence':'重点问咳嗽'},
                       {'problem_id':'t1:f1','operation':'withdraw','evidence':'不重点问头疼了'}]})
        agent.fact_extractor.intent=ComplaintIntent(client,prompt_version='v2')
        agent.run('头痛和咳嗽，主要想问头痛。')
        agent.run('改成重点问咳嗽，不重点问头疼了。')
        state=agent.memory.get_case_state()
        self.assertEqual(state['chief_complaints'],['咳嗽'])
        self.assertEqual(set(state['symptoms']),{'头痛','咳嗽'})
        self.assertEqual(state['current_focus'],'t1:f2')
        self.assertEqual(state['complaint_intent_history'][-1]['version'],'complaint_intent_v2')

    def test_intent_selection_withdrawal_and_fact_correction(self):
        agent, _ = self.agent(delta(fact('头痛', '主要想问头痛'), fact('咳嗽', id='f2')), delta(),
            delta(fact('头痛', '没有头痛', polarity='negated', operation='correct', replaces='t1:f1')))
        client = FixedClient({'events': [{'problem_id': 't1:f1', 'operation': 'select', 'evidence': '主要想问头痛'}]},
            {'events': [{'problem_id': 't1:f1', 'operation': 'withdraw', 'evidence': '不重点问头痛了'}]}, {'events': []})
        agent.fact_extractor.intent = ComplaintIntent(client)
        agent.run('还有咳嗽，主要想问头痛。')
        state = agent.memory.get_case_state()
        self.assertEqual(state['chief_complaints'], ['头痛'])
        self.assertEqual(state['chief_complaint_candidates'], ['咳嗽'])
        self.assertEqual(state['current_focus'], 't1:f1')
        agent.run('不重点问头痛了。')
        state = agent.memory.get_case_state()
        self.assertEqual(state['chief_complaints'], [])
        self.assertIn('头痛', state['symptoms'])
        agent.run('刚才说错了，没有头痛。')
        self.assertEqual(agent.memory.get_case_state()['chief_complaint_candidates'], ['咳嗽'])

    def test_high_skips_intent_provider(self):
        agent, _ = self.agent(delta(fact('呼吸困难', field='red_flags')))
        client = FixedClient()
        agent.fact_extractor.intent = ComplaintIntent(client)
        response = agent.run('呼吸困难。')
        self.assertFalse(client.calls)
        self.assertIn('需要优先警惕', response)
        self.assertEqual(agent.memory.get_case_state()['intent_status'], 'skipped')

    def test_skipped_intent_preserves_fact_failure(self):
        agent, _ = self.agent(delta(fact('头痛', '不存在的证据')))
        agent.fact_extractor.intent = ComplaintIntent(FixedClient())
        response = agent.run('头痛。')
        self.assertTrue(agent.memory.get_case_state()['fact_extraction_error'])
        self.assertIn('未能可靠识别', response)

    def test_invalid_intent_does_not_mutate_facts(self):
        agent, _ = self.agent(delta(fact('头痛')))
        agent.fact_extractor.intent = ComplaintIntent(FixedClient({'events': [
            {'problem_id': 'invented', 'operation': 'select', 'evidence': '头痛'}]}))
        agent.run('头痛。')
        state = agent.memory.get_case_state()
        self.assertEqual(state['intent_status'], 'error')
        self.assertEqual(state['symptoms'], ['头痛'])
        self.assertEqual(state['chief_complaints'], [])


class IntentContractTests(unittest.TestCase):
    def test_v3_is_opt_in_and_v2_prompt_is_preserved(self):
        from agent.complaint_intent import PROMPTS
        self.assertTrue(PROMPTS['v3'].startswith(PROMPTS['v2']))
        self.assertEqual(ComplaintIntent(FixedClient()).prompt_version,'v1')
        self.assertEqual(ComplaintIntent(FixedClient(),prompt_version='v3').prompt_version,'v3')

    def test_multiple_intents_and_context(self):
        inp = {'current_turn': '头痛和咳嗽', 'last_question': '本次主要想咨询什么？',
               'problems': [{'id': 'a'}, {'id': 'b'}]}
        events = [{'problem_id': id_, 'operation': 'select', 'evidence': evidence}
                  for id_, evidence in [('a', '头痛'), ('b', '咳嗽')]]
        self.assertEqual(validate_events(inp, {'events': events}), events)
        for invalid in ({'events': events + events}, {'events': [dict(events[0], evidence='不存在')]},
                        {'events': [dict(events[0], operation='deny')]}):
            with self.assertRaises(ValueError):
                validate_events(inp, invalid)
