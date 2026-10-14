from copy import deepcopy
import unittest
from evaluation.reassertion_metrics import score_reassertions, project_reassertions
from llm.fact_contract_runtime import V2, build_input
from training.test_fact_contract import fact
from memory.fact_ledger_runtime import empty_ledger, next_input, apply_delta, effective_records


class ReassertionTests(unittest.TestCase):
    def setup_case(self, text='腹痛的程度很轻。'):
        ledger=empty_ledger(contract_version=V2)
        inp=next_input(ledger,'我有腹痛。')
        ledger=apply_delta(ledger,inp,{'facts':[fact('腹痛','我有腹痛')]})
        inp=next_input(ledger,text)
        gold={'facts':[fact('轻度',text[:-1],field='severity',about='t1:f1')]}
        pred={'facts':[fact('腹痛'),fact('轻度',text[:-1],id='f2',field='severity',about='f1')]}
        return ledger,inp,gold,pred

    def test_same_object_binding_without_rewriting_raw(self):
        _,inp,gold,pred=self.setup_case()
        before=deepcopy(pred)
        result=score_reassertions(pred,gold,inp)
        self.assertFalse(result['original']['raw']['semantics_exact'])
        self.assertTrue(result['explicit_reassertion_equivalent']['operations_exact'])
        self.assertEqual(pred,before)

    def test_ledger_already_keeps_raw_records_and_single_effective_identity(self):
        ledger,inp,_,pred=self.setup_case()
        updated=apply_delta(ledger,inp,pred)
        self.assertEqual(len(updated['records']),3)
        self.assertEqual(updated['aliases']['t2:f1'],'t1:f1')
        effective=effective_records(updated)
        self.assertEqual(len(effective),2)
        self.assertEqual(next(f for f in effective if f['field']=='severity')['about'],'t1:f1')

    def test_defer_hypothesis_questions_and_other_scopes_not_erased(self):
        for text in ['腹痛以后再说。','如果腹痛出现再说。','我妈妈有腹痛。','腹痛是你说的吗？','也许我腹痛。']:
            _,inp,_,_=self.setup_case(text)
            pred={'facts':[fact('腹痛')]}
            transformed,changes=project_reassertions(pred,inp)
            self.assertEqual(transformed,pred,text)
            self.assertEqual(changes,[],text)

    def test_different_polarity_subject_time_and_operations_not_erased(self):
        _,inp,_,_=self.setup_case()
        for key,value in [('polarity','negated'),('subject','other'),('temporality','hypothetical')]:
            pred={'facts':[fact('腹痛')]};pred['facts'][0][key]=value
            self.assertEqual(project_reassertions(pred,inp)[0],pred)
        pred={'facts':[fact('腹痛')]};pred['facts'][0].update(operation='correct',replaces='t1:f1')
        self.assertEqual(project_reassertions(pred,inp)[0],pred)

    def test_missing_new_facts_and_wrong_binding_remain_failures(self):
        _,inp,gold,_=self.setup_case()
        result=score_reassertions({'facts':[fact('腹痛')]},gold,inp)
        self.assertFalse(result['explicit_reassertion_equivalent']['semantics_exact'])

    def test_invalid_evidence_not_removed(self):
        _,inp,gold,pred=self.setup_case()
        pred['facts'][0]['evidence']='腹部疼痛'
        result=score_reassertions(pred,gold,inp)
        self.assertFalse(result['explicit_reassertion_equivalent']['contract_valid'])


if __name__=='__main__':unittest.main()
