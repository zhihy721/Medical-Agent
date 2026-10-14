"""Deterministic family shuffle and optimizer-boundary resume schedule.

This schedules rows only. A real trainer must checkpoint model/optimizer/scheduler/RNG
at each committed optimizer boundary and replay an unfinished window after crash.
"""
from collections import defaultdict
import hashlib
import json
import random


def data_signature(rows):
    fields=[(r['sample_id'],r['source_family_id'],r['split'],r.get('input'),r.get('target'))
            for r in rows]
    return hashlib.sha256(json.dumps(fields,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def schedule(rows, *, seed=42, epochs=1, accumulation=8):
    if not 1<=epochs<=10 or not 1<=accumulation<=64:
        raise ValueError('invalid training schedule configuration')
    if not rows or len({r['sample_id'] for r in rows})!=len(rows):
        raise ValueError('empty or duplicate training rows')
    if any(r['split']!='train' for r in rows):
        raise ValueError('validation row in training schedule')
    families=defaultdict(list)
    for row in rows:families[row['source_family_id']].append(row['sample_id'])
    windows=[]
    for epoch in range(epochs):
        rng=random.Random(seed+epoch)
        names=sorted(families)
        rng.shuffle(names)
        order=[]
        for family in names:
            siblings=sorted(families[family])
            rng.shuffle(siblings)
            order.extend(siblings)
        for start in range(0,len(order),accumulation):
            ids=order[start:start+accumulation]
            windows.append(dict(epoch=epoch,step=len(windows)+1,sample_ids=ids,
                                loss_divisor=len(ids),samples_before=epoch*len(rows)+start,
                                samples_after=epoch*len(rows)+start+len(ids)))
    return dict(data_signature=data_signature(rows),seed=seed,epochs=epochs,
                accumulation=accumulation,windows=windows,
                expected_samples=len(rows)*epochs,expected_optimizer_steps=len(windows))


def resume_windows(plan, committed_step, *, data_signature_value):
    if plan['data_signature']!=data_signature_value:
        raise ValueError('resume data mismatch')
    if type(committed_step) is not int or not 0<=committed_step<=plan['expected_optimizer_steps']:
        raise ValueError('invalid committed optimizer step')
    completed=plan['windows'][:committed_step]
    remaining=plan['windows'][committed_step:]
    accounted=completed[-1]['samples_after'] if completed else 0
    if remaining and remaining[0]['samples_before']!=accounted:
        raise ValueError('schedule cursor mismatch')
    return dict(samples_committed=accounted,next_windows=remaining)


def main():
    import argparse
    from pathlib import Path
    from training.teacher_data import read_rows, sha
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--accumulation',type=int,default=8)
    a=p.parse_args()
    train=[r for r in read_rows(a.data) if r['split']=='train']
    plan=schedule(train,seed=a.seed,epochs=a.epochs,accumulation=a.accumulation)
    plan.update(scope='schedule_only_not_training',source_data_sha256=sha(a.data),
                script_sha256=sha(__file__))
    with a.out.open('x',encoding='utf-8') as stream:
        json.dump(plan,stream,ensure_ascii=False,indent=2)
    print(json.dumps({'train_rows':len(train),'epochs':a.epochs,
                      'optimizer_steps':plan['expected_optimizer_steps'],
                      'samples':plan['expected_samples']}))


if __name__=='__main__':main()
