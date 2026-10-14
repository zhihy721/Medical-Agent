import unittest
from agent import test_semantic_runtime as runtime_tests
from agent.test_semantic_runtime import delta
from training.test_fact_contract import fact


class CorrectionNoticeTests(unittest.TestCase):
    setUp=runtime_tests.SemanticRuntimeTests.setUp
    agent=runtime_tests.SemanticRuntimeTests.agent

    def test_explicit_correction_acknowledged_and_risk_reevaluated(self):
        agent,_=self.agent(delta(fact('呼吸困难',field='red_flags'),fact('腰痛',id='f2')),
            delta(fact('呼吸困难','没有呼吸困难',field='red_flags',polarity='negated',
                operation='correct',replaces='t1:f1')))
        agent.run('呼吸困难，腰痛。')
        reply=agent.run('刚才讲错了，没有呼吸困难。')
        self.assertIn('已记录你对先前描述的更正',reply)
        self.assertEqual(agent.memory.get_case_state()['risk_level'],'LOW')

    def test_true_resolution_is_not_reported_as_a_correction(self):
        agent,_=self.agent(delta(fact('呼吸困难',field='red_flags')),
            delta(fact('呼吸困难','现在缓解了',field='red_flags',temporality='resolved',
                operation='resolve',replaces='t1:f1')))
        agent.run('呼吸困难。');reply=agent.run('现在缓解了。')
        self.assertNotIn('对先前描述的更正',reply)
        self.assertEqual(agent.memory.get_case_state()['risk_level'],'HIGH')


if __name__=='__main__':unittest.main()
