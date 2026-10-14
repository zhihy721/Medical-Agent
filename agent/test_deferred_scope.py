"""Real graph checks for deferral uncertainty and safe confirmation routing."""
import unittest
from agent import test_semantic_runtime as runtime_fixture
from training.test_fact_contract import fact

delta=runtime_fixture.delta
from memory.fact_ledger import project,risk_input
from llm.fact_contract import build_input
from llm.deferred_scope import adjust_delta,is_deferred_ambiguous
from agent.safety_gate import apply_proposal


class DeferredScopeTests(unittest.TestCase):
    def setUp(self):
        self.runtime=runtime_fixture.SemanticRuntimeTests()
        self.runtime.setUp()
        self.addCleanup(self.runtime.doCleanups)

    def test_plain_deferral_is_uncertain_and_actual_condition_stays_hypothetical(self):
        inp=build_input('耳痛的事以后再说。')
        raw=delta(fact('耳痛','耳痛的事以后再说',temporality='hypothetical'))
        effective,changes=adjust_delta(inp,raw)
        self.assertEqual(effective['facts'][0]['polarity'],'uncertain')
        self.assertEqual(effective['facts'][0]['temporality'],'current')
        self.assertEqual(raw['facts'][0]['polarity'],'present')
        self.assertEqual(len(changes),1)
        for text in ['我现在耳痛，不过耳痛以后再说','耳朵现在疼，回头再聊']:
            d=delta(fact('耳痛',text))
            self.assertEqual(adjust_delta(build_input(text),d),(d,[]))
        for text in ['如果以后耳痛再说','耳痛以后要是出现了再说','假设以后耳痛再说','我确实耳痛，耳痛以后再说']:
            with self.subTest(text=text):
                self.assertFalse(is_deferred_ambiguous(text,text))

    def test_uncertainty_is_recorded_and_existence_question_precedes_attributes(self):
        agent,_=self.runtime.agent(delta(fact('耳痛','耳痛的事以后再说',temporality='hypothetical')),
                                  delta(fact('耳痛','我没有耳痛',polarity='negated')))
        reply=agent.run('耳痛的事以后再说。')
        state=agent.memory.get_case_state()
        self.assertEqual(state['fact_view']['facts'][0]['polarity'],'uncertain')
        self.assertNotIn('耳痛',state['symptoms'])
        self.assertIn('你现在有耳痛吗',reply)
        trace=agent.get_graph_state().values['extraction_trace']
        self.assertEqual(trace['raw_delta']['facts'][0]['polarity'],'present')
        self.assertEqual(len(trace['scope_adjustments']),1)
        self.assertEqual(state['fact_ledger']['turns'][0]['extraction_policy_trace']['extractor_raw_delta'],trace['raw_delta'])
        agent.run('我没有耳痛，只是想以后讨论。')
        state=agent.memory.get_case_state()
        self.assertFalse(state['fact_clarifications'])
        self.assertFalse(state['contradictions'])
        self.assertEqual(state['fact_view']['facts'][0]['polarity'],'negated')

    def test_deferral_does_not_retract_known_positive_and_high_remains_first(self):
        agent,_=self.runtime.agent(delta(fact('呼吸困难',field='red_flags')),
                 delta(fact('耳痛','耳痛的事以后再说',temporality='hypothetical')))
        agent.run('呼吸困难。')
        reply=agent.run('耳痛的事以后再说。')
        state=agent.memory.get_case_state()
        self.assertEqual(state['risk_level'],'HIGH')
        self.assertIn('需要优先警惕',reply)
        self.assertNotIn('你现在有耳痛吗',reply)
        agent,_=self.runtime.agent(delta(fact('耳痛')),delta(fact('耳痛','耳痛的事以后再说')))
        agent.run('耳痛。');agent.run('耳痛的事以后再说。')
        state=agent.memory.get_case_state()
        self.assertIn('耳痛',state['symptoms'])
        self.assertFalse(state['fact_clarifications'])

    def test_model_plan_cannot_skip_confirmation(self):
        case=dict(fact_clarifications=[dict(fact_id='t1:f1',question='你现在有耳痛吗')])
        plan,audit=apply_proposal({'next_action':'final_advice'}, {'next_action':'final_advice'},case,{'risk':'LOW'})
        self.assertEqual(plan['next_action'],'ask_followup_single')
        self.assertEqual(audit['rule_override_reason'],'uncertain_deferred_symptom')

    def test_confirmation_repeats_at_most_twice_and_no_fact_is_fabricated(self):
        agent,_=self.runtime.agent(delta(fact('耳痛','耳痛的事以后再说',temporality='hypothetical')),delta(),delta())
        replies=[agent.run(t) for t in ('耳痛的事以后再说。','嗯。','嗯。')]
        self.assertIn('你现在有耳痛吗',replies[0]);self.assertIn('你现在有耳痛吗',replies[1])
        self.assertNotIn('你现在有耳痛吗',replies[2])
        inp=build_input('耳痛的事以后再说。')
        self.assertEqual(adjust_delta(inp,delta())[0],delta())


if __name__=='__main__':unittest.main()
