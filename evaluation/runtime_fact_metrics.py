"""V2 scoring seam; locked V1 scores and datasets remain untouched."""
from copy import deepcopy
import json
from evaluation.fact_metrics import signatures
from llm.fact_contract_runtime import V2, position_delta
from llm.fact_representation import normalize_representation
from memory.fact_identity import same_concept, VERSION as IDENTITY_VERSION

VERSION='runtime_fact_metrics_v1'


def _score(pred,gold,inp,*,validated=False):
    if not validated:position_delta(gold,inp)
    result=dict(contract_valid=False,semantics_exact=False,bindings_exact=False,
        operations_exact=False,evidence_exact=False,strict_success=False,semantic_counts=None,error='')
    try:
        if not validated:position_delta(pred,inp)
        result['contract_valid']=True
        p,g=signatures(pred,'semantics'),signatures(gold,'semantics')
        result['semantic_counts']=dict(matched=sum((p & g).values()),predicted=sum(p.values()),gold=sum(g.values()))
        for key,level in [('semantics_exact','semantics'),('bindings_exact','bindings'),
                          ('operations_exact','operations'),('evidence_exact','evidence')]:
            result[key]=signatures(pred,level)==signatures(gold,level)
        result['strict_success']=result['evidence_exact']
    except (ValueError,KeyError,TypeError) as exc:result['error']=str(exc)
    return result


def _identity(delta):
    output=deepcopy(delta)
    for f in output['facts']:
        reference=dict(f,field='red_flags',value='呼吸困难')
        if same_concept(f,reference,version=IDENTITY_VERSION):f['field']='red_flags'
    return output


def score_v2(prediction,gold,inp):
    """Raw, represented, same-concept scores are separate, never overwrite raw.

    Gold must be valid. No inferred facts or corrected bindings are supplied to
    a model. Concept scoring validates before field projection; it cannot rescue
    an invalid replacement, missing evidence, polarity, subject or time error.
    """
    if inp['contract_version']!=V2:raise ValueError('V2 scorer requires V2 input')
    position_delta(gold,inp)
    try:pred=json.loads(prediction) if isinstance(prediction,str) else prediction
    except (ValueError,TypeError):pred=None
    raw=_score(pred,gold,inp)
    canonical_gold,gold_changes=normalize_representation(gold,inp)
    position_delta(canonical_gold,inp)
    try:
        canonical_pred,changes=normalize_representation(pred,inp)
        represented=_score(canonical_pred,canonical_gold,inp)
    except (ValueError,KeyError,TypeError) as exc:
        canonical_pred=None;changes=[];represented=_score(None,canonical_gold,inp)
        represented['error']=str(exc)
    if represented['contract_valid']:
        concept=_score(_identity(canonical_pred),_identity(canonical_gold),inp,validated=True)
    else:concept=deepcopy(represented)
    return dict(version=VERSION,raw=raw,represented=represented,concept_equivalent=concept,
        normalization_changes=changes,gold_normalization_changes=gold_changes,
        identity_version=IDENTITY_VERSION,scope='single_fixed_input_not_whole_agent_or_clinical_quality')
