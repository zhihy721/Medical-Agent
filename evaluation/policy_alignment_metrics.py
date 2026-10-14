"""Extra append-only policy-equivalence score; original experiment stays fixed."""
import json
from evaluation.reassertion_metrics import score_reassertions
from llm.fact_policy_alignment import align,VERSION


def score_policy(pred,gold,inp):
    original=score_reassertions(pred,gold,inp)
    gold_aligned,context,gold_changes=align(gold,inp)
    changes=[]
    try:
        parsed=json.loads(pred) if isinstance(pred,str) else pred
        aligned,pcontext,changes=align(parsed,inp)
        if context!=pcontext:raise ValueError('context conversion differs')
        result=score_reassertions(aligned,gold_aligned,context)
        metrics=result['explicit_reassertion_equivalent']
    except (ValueError,TypeError,KeyError):
        metrics=original['explicit_reassertion_equivalent']
    return dict(version=VERSION,original=original,policy_equivalent=metrics,
                prediction_changes=changes,gold_changes=gold_changes,
                scope='finite_field_and_full_value_equivalence_not_medical_entailment')
