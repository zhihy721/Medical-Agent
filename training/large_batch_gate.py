"""User-requested size extension, bound to remaining quotas; legacy lock unchanged."""
from collections import Counter
import json
from pathlib import Path

from training.formal_batch_gate import verify_lock
from training.holdout_reservation import verify_reserved_holdout
from training.teacher_data import read_rows, sha


def verify_extension(plan, pool):
    extension=plan.get('large_batch_extension',{})
    if extension.get('status')!='user_requested_large_batch_v1':
        raise ValueError('large batch requires size extension')
    if extension.get('legacy_lock_sha256')!=sha(plan['contract_lock']):
        raise ValueError('large batch legacy lock differs')
    verify_lock(plan['contract_lock'])
    maximum=extension.get('max_candidate_count')
    if type(maximum) is not int or not 80<maximum<=1000 or plan['candidate_count']>maximum:
        raise ValueError('invalid large batch capacity')
    policy=json.loads(Path('training/formal_data_plan_v1.json').read_text(encoding='utf-8'))
    current=Counter((r['split'],r['primary_stratum']) for r in pool)
    targets={s['name']:s for s in policy['primary_strata']}
    names=[s['name'] for s in plan['primary_strata']]
    if len(names)!=len(set(names)) or set(names)!=set(targets):
        raise ValueError('large batch strata differ')
    for s in plan['primary_strata']:
        for split in ('train','validation'):
            n=s[split]
            if type(n) is not int or n<0 or n+current[split,s['name']]>targets[s['name']][split]:
                raise ValueError('large batch exceeds remaining stratum quota')
    if sum(s[split] for s in plan['primary_strata'] for split in ('train','validation'))!=plan['candidate_count']:
        raise ValueError('large batch count differs from quotas')
    return maximum


def authorize(plan_path, prepared, out):
    plan_path,prepared,out=map(Path,(plan_path,prepared,out))
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    pool=read_rows(plan['existing_reviewed_records'])
    if sha(plan['existing_reviewed_records'])!=plan['pool_sha256']:
        raise ValueError('large batch pool changed')
    limit=verify_extension(plan,pool)
    preparation=json.loads((prepared/'audit.json').read_text(encoding='utf-8'))
    if preparation['plan_sha256']!=sha(plan_path) or not preparation['ready_for_generation']:
        raise ValueError('large preparation stale')
    cards=read_rows(prepared/'cards.jsonl')
    if len(cards)!=plan['candidate_count'] or preparation['source_cards_sha256']!=sha(prepared/'cards.jsonl'):
        raise ValueError('large prepared count differs')
    verify_reserved_holdout(read_rows(prepared/'registry.jsonl'))
    result=dict(status='user_requested_large_candidate_batch_only',training_release_allowed=False,
        max_candidate_count=limit,plan=str(plan_path),plan_sha256=sha(plan_path),
        cards=str(prepared/'cards.jsonl'),cards_sha256=sha(prepared/'cards.jsonl'),
        registry_sha256=sha(prepared/'registry.jsonl'),legacy_lock_sha256=sha(plan['contract_lock']),
        script_sha256=sha(__file__))
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    return result


def verify_authorization(path, cards, registry, lock):
    grant=json.loads(Path(path).read_text(encoding='utf-8'))
    if grant['status']!='user_requested_large_candidate_batch_only' or grant['training_release_allowed'] is not False:
        raise ValueError('invalid large candidate authorization')
    if grant['script_sha256']!=sha(__file__) or grant['plan_sha256']!=sha(grant['plan']):
        raise ValueError('large candidate authorization stale')
    plan=json.loads(Path(grant['plan']).read_text(encoding='utf-8'))
    if sha(plan['existing_reviewed_records'])!=plan['pool_sha256']:
        raise ValueError('large authorization pool changed')
    limit=verify_extension(plan,read_rows(plan['existing_reviewed_records']))
    if grant['legacy_lock_sha256']!=sha(lock) or grant['registry_sha256']!=sha(registry):
        raise ValueError('large authorization registry/lock differs')
    if grant['cards_sha256']!=sha(grant['cards']) or grant['max_candidate_count']!=limit:
        raise ValueError('large authorization cards/capacity differ')
    allowed={r['sample_id']:r for r in read_rows(grant['cards'])}
    if not cards or len(cards)>limit or any(allowed.get(r['sample_id'])!=r for r in cards):
        raise ValueError('large run has unauthorized cards')
    return dict(sha256=sha(path),max_candidate_count=limit,training_release_allowed=False)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('plan','prepared','out'):p.add_argument('--'+n,required=True)
    a=p.parse_args();r=authorize(a.plan,a.prepared,a.out)
    print(json.dumps({'candidate_limit':r['max_candidate_count'],'training_release_allowed':False}))
