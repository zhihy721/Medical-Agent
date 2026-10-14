from copy import deepcopy
import unittest
from agent import test_representation_runtime as representation_tests
from training.test_fact_contract import fact
from agent.graph import _run_risk_assessment


class TrajectoryRepairTests(unittest.TestCase):
    def extract(self,text,*facts,memory=None):
        return representation_tests.RepresentationTests().run_fact(text,*facts,memory=memory)

    def test_identical_affirmed_dyspnea_has_identical_high_policy(self):
        for field in ['symptoms','red_flags']:
            m,a=self.extract('现在呼吸困难。',fact('呼吸困难',field=field))
            self.assertEqual(a['status'],'ok')
            before=deepcopy(m.case_state['fact_ledger'])
            self.assertEqual(_run_risk_assessment(m.case_state)['risk'],'HIGH')
            self.assertEqual(before,m.case_state['fact_ledger'])

    def test_resolved_symptom_representation_retains_safety(self):
        m,_=self.extract('现在呼吸困难。',fact('呼吸困难'))
        m,a=self.extract('刚才呼吸困难已缓解。',fact('呼吸困难',field='symptoms',temporality='resolved',
            operation='resolve',replaces='t1:f1'),memory=m)
        self.assertEqual(a['status'],'ok')
        self.assertEqual(_run_risk_assessment(m.case_state)['risk'],'HIGH')

    def test_scoped_or_denied_dyspnea_does_not_upgrade(self):
        for overrides in [dict(polarity='negated'),dict(temporality='hypothetical'),dict(subject='other')]:
            m,a=self.extract('如果有呼吸困难再联系。',fact('呼吸困难',**overrides))
            self.assertEqual(a['status'],'ok')
            self.assertNotEqual(_run_risk_assessment(m.case_state)['risk'],'HIGH')

    def test_deferring_known_sleep_keeps_fact_and_duration(self):
        m,_=self.extract('多梦八天。',fact('多梦',field='sleep'),fact('八天',field='duration',id='f2',about='f1'))
        m,a=self.extract('多梦的事情先放一放。',fact('多梦',field='sleep',polarity='uncertain'),memory=m)
        self.assertEqual(a['status'],'ok')
        self.assertEqual(m.case_state['sleep'],'多梦')
        self.assertEqual(m.case_state['contradictions'],[])
        self.assertEqual(m.case_state['observation_attributes'][0]['attributes'][0]['value'],'八天')

    def test_other_subject_later_does_not_hide_own_denial_omission(self):
        m,a=self.extract('我没有发热，也没有畏寒。父亲失眠。',
            fact('失眠','父亲失眠',field='sleep',subject='other'))
        self.assertEqual(a['status'],'error')
        self.assertIn('denial',a['error'])

    def test_other_and_conditional_denials_do_not_need_patient_labels(self):
        for text in ['父亲没有失眠，也没有发热。我咳嗽两天。',
                     '如果以后咳嗽，没有发热就再讨论。我现在咳嗽两天。']:
            m,a=self.extract(text,fact('咳嗽','我现在咳嗽两天' if '现在' in text else '我咳嗽两天'),
                fact('两天','我现在咳嗽两天' if '现在' in text else '我咳嗽两天',field='duration',id='f2',about='f1'))
            self.assertEqual(a['status'],'ok',a.get('error'))
