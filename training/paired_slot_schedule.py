"""Validate preregistered paired slots; never replace them with family shuffling."""
import random
try:
    from .formal_schedule import data_signature
except ImportError:
    from formal_schedule import data_signature


def verify_schedule(rows, plan, seed=42):
    if len(rows)!=280 or len({r['sample_id'] for r in rows})!=280:
        raise ValueError('280 unique samples required')
    if any(r['split']!='train' or r.get('paired_slot')!=i for i,r in enumerate(rows)):
        raise ValueError('train-only paired slots required')
    if len({r['source_family_id'] for r in rows})!=280:
        raise ValueError('280 independent families required')
    expected=dict(version='paired_slot_schedule_v1',data_signature=data_signature(rows),
        seed=seed,epochs=1,accumulation=8,expected_samples=280,expected_optimizer_steps=35)
    if any(plan.get(k)!=v for k,v in expected.items()):raise ValueError('schedule protocol mismatch')
    order=list(range(280));random.Random(seed).shuffle(order)
    if len(plan['windows'])!=35:raise ValueError('35 windows required')
    for i,w in enumerate(plan['windows']):
        slots=order[i*8:i*8+8]
        actual=dict(step=i+1,epoch=0,paired_slots=slots,sample_ids=[rows[s]['sample_id'] for s in slots],
            loss_divisor=8,samples_before=i*8,samples_after=(i+1)*8)
        if w!=actual:raise ValueError('paired window/order mismatch')
    return plan
