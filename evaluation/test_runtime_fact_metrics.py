from copy import deepcopy
import unittest
from evaluation.runtime_fact_metrics import score_v2
from llm.fact_contract_runtime import V2,build_input
from training.test_fact_contract import fact


class RuntimeMetricsTests(unittest.TestCase):
    def inp(self,text):return build_input(text,contract_version=V2)

    def test_sleep_duration_supported_without_touching_v1_scorer(self):
        from evaluation.fact_metrics import score_delta
        inp=self.inp('多梦九天。');gold={'facts':[fact('多梦',field='sleep'),fact('九天',id='f2',field='duration',about='f1')]}
        with self.assertRaises(ValueError):score_delta(gold,gold,inp)
        self.assertTrue(score_v2(gold,gold,inp)['raw']['strict_success'])

    def test_wrong_binding_is_not_rescued_by_semantic_or_field_aliases(self):
        inp=self.inp('头痛三天，牙痛。');gold={'facts':[fact('头痛'),fact('三天',id='f2',field='duration',about='f1'),fact('牙痛',id='f3')]}
        pred=deepcopy(gold);pred['facts'][1]['about']='f3'
        result=score_v2(pred,gold,inp)
        for stage in ['raw','represented','concept_equivalent']:
            self.assertTrue(result[stage]['semantics_exact']);self.assertFalse(result[stage]['bindings_exact'])

    def test_identity_fields_are_separate_scores_and_never_polarity_or_scope_conversion(self):
        inp=self.inp('没有呼吸困难。');gold={'facts':[fact('呼吸困难','没有呼吸困难',field='red_flags',polarity='negated')]}
        pred=deepcopy(gold);pred['facts'][0]['field']='symptoms'
        result=score_v2(pred,gold,inp)
        self.assertFalse(result['raw']['operations_exact']);self.assertFalse(result['represented']['operations_exact'])
        self.assertTrue(result['concept_equivalent']['operations_exact'])
        for key,value in [('subject','other'),('temporality','hypothetical'),('polarity','present')]:
            bad=deepcopy(pred);bad['facts'][0][key]=value
            self.assertFalse(score_v2(bad,gold,inp)['concept_equivalent']['semantics_exact'])

    def test_synonym_preserves_raw_score_and_output_bytes(self):
        inp=self.inp('头疼。');gold={'facts':[fact('头痛','头疼')]};pred={'facts':[fact('头疼','头疼')]}
        before=deepcopy(pred);r=score_v2(pred,gold,inp)
        self.assertFalse(r['raw']['semantics_exact']);self.assertTrue(r['represented']['strict_success'])
        self.assertEqual(pred,before)

    def test_missing_fact_bad_evidence_and_malformed_json_remain_failures(self):
        inp=self.inp('头痛。');gold={'facts':[fact('头痛')]}
        for pred in ['not json',None,{'facts':[]},{'facts':[fact('头痛','原文没有这个')]}]:
            r=score_v2(pred,gold,inp)
            self.assertFalse(r['concept_equivalent']['strict_success'])

    def test_invalid_gold_rejected_instead_of_inflating_denominator(self):
        with self.assertRaises(ValueError):score_v2({}, {'facts':[fact('牙痛')]},self.inp('头痛。'))

    def test_identity_projection_does_not_revalidate_as_a_model_output_or_drop_counts(self):
        inp=self.inp('呼吸困难，喘不上气。')
        gold={'facts':[fact('呼吸困难','呼吸困难',field='red_flags'),
            fact('呼吸困难','喘不上气',field='symptoms',id='f2')]}
        r=score_v2(gold,gold,inp)
        self.assertTrue(r['concept_equivalent']['strict_success'])
        self.assertEqual(r['concept_equivalent']['semantic_counts']['gold'],2)
