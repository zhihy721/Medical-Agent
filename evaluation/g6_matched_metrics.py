"""Fixed G6 views and finite rejection checks; failed gold stays in denominators."""
from copy import deepcopy
import json
from evaluation.fact_metrics import signatures
from evaluation.runtime_fact_metrics import _score
from evaluation.reassertion_metrics import project_reassertions
from llm.fact_contract_runtime import position_delta
from llm.fact_projection_policy import ALIAS_VERSION
from llm.fact_representation import normalize_representation
from llm.linked_fact_guard_v16 import check
from llm.runtime_extraction_guard import classify

VERSION='g6_matched_counts_v1'
VIEWS=('raw','finite_v5')
LEVELS=('semantics','bindings','operations','evidence')
FLAGS=('contract_valid','semantics_exact','bindings_exact','operations_exact','evidence_exact')


def project(delta,inp,view):
    if view=='raw':position_delta(delta,inp);return deepcopy(delta)
    converted,_=normalize_representation(delta,inp,alias_version=ALIAS_VERSION)
    position_delta(converted,inp)
    return project_reassertions(converted,inp)[0]


def score(pred,gold,inp):
    position_delta(gold,inp)
    try:parsed=json.loads(pred) if isinstance(pred,str) else pred
    except (ValueError,TypeError):parsed=None
    attempted=len(parsed['facts']) if isinstance(parsed,dict) and isinstance(parsed.get('facts'),list) else 0
    views={}
    for view in VIEWS:
        g=project(gold,inp,view)
        try:
            p=project(parsed,inp,view);flags=_score(p,g,inp,validated=True)
            counts={level:dict(matched=sum((signatures(p,level)&signatures(g,level)).values()),
                predicted=len(p['facts']),gold=len(g['facts'])) for level in LEVELS}
        except (ValueError,KeyError,TypeError) as exc:
            flags=dict.fromkeys(FLAGS,False);flags['error']=str(exc)
            counts={level:dict(matched=0,predicted=attempted,gold=len(g['facts'])) for level in LEVELS}
        views[view]=dict(flags=flags,counts=counts)
    error=''
    try:
        converted,_=normalize_representation(parsed,inp,alias_version=ALIAS_VERSION)
        check(inp,converted,classify(inp['current_turn']))
    except (ValueError,KeyError,TypeError) as exc:error=str(exc)
    return dict(version=VERSION,views=views,guard_pass=not error,guard_error=error,
        first_success=bool(views['finite_v5']['flags']['operations_exact'] and not error))


def summarize(records):
    result=dict(n=len(records),views={},guard_pass=sum(r['scores']['guard_pass'] for r in records),
        first_success=sum(r['scores']['first_success'] for r in records))
    for view in VIEWS:
        sub=[r['scores']['views'][view] for r in records]
        entry=dict(exact={k:sum(bool(r['flags'][k]) for r in sub) for k in FLAGS},facts={})
        for level in LEVELS:
            sums={k:sum(r['counts'][level][k] for r in sub) for k in ('matched','predicted','gold')}
            m,p,g=[sums[k] for k in ('matched','predicted','gold')]
            entry['facts'][level]=dict(**sums,precision=m/p if p else 0.,recall=m/g if g else 0.,f1=2*m/(p+g) if p+g else 0.)
        result['views'][view]=entry
    return result
