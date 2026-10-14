from copy import deepcopy
import unittest
from llm.fact_policy_alignment import align
from evaluation.policy_alignment_metrics import score_policy
from llm.fact_contract_runtime import V2,build_input
from training.test_fact_contract import fact


class AlignmentTests(unittest.TestCase):
    def test_exact_thirst_field_keeps_all_fact_dimensions_and_original(self):
        inp=build_input('我口渴。',contract_version=V2)
        pred={'facts':[fact('口渴','我口渴',field='thirst')]};before=deepcopy(pred)
        gold={'facts':[fact('口渴','我口渴')]}
        r=score_policy(pred,gold,inp)
        self.assertFalse(r['original']['original']['raw']['semantics_exact'])
        self.assertTrue(r['policy_equivalent']['operations_exact'])
        self.assertEqual(pred,before)
        for key,value in [('polarity','negated'),('subject','other'),('temporality','hypothetical')]:
            wrong=deepcopy(pred);wrong['facts'][0][key]=value
            self.assertFalse(score_policy(wrong,gold,inp)['policy_equivalent']['semantics_exact'])

    def test_full_registered_values_only_not_fuzzy_inference(self):
        inp=build_input('嗓子疼，鼻子不通气。',contract_version=V2)
        p={'facts':[fact('嗓子疼'),fact('鼻子不通气',id='f2')]}
        result,_,_=align(p,inp)
        self.assertEqual([r['value'] for r in result['facts']],['咽痛','鼻塞'])
        for field,value in [('thirst','口干'),('appetite','吃得少'),('sweating','出汗'),
                             ('symptoms','胃痛'),('symptoms','黑便'),('symptoms','胸痛')]:
            original={'facts':[fact(value,field=field)]}
            self.assertEqual(align(original,inp)[0],original)

    def test_bad_evidence_and_wrong_relation_remain_wrong(self):
        inp=build_input('嗓子疼三天，牙痛。',contract_version=V2)
        gold={'facts':[fact('咽痛','嗓子疼'),fact('三天',id='f2',field='duration',about='f1'),fact('牙痛',id='f3')]}
        wrong=deepcopy(gold);wrong['facts'][0]['value']='嗓子疼';wrong['facts'][1]['about']='f3'
        r=score_policy(wrong,gold,inp)
        self.assertTrue(r['policy_equivalent']['semantics_exact']);self.assertFalse(r['policy_equivalent']['bindings_exact'])
        wrong['facts'][0]['evidence']='喉咙不舒服'
        self.assertFalse(score_policy(wrong,gold,inp)['policy_equivalent']['contract_valid'])


if __name__=='__main__':unittest.main()
