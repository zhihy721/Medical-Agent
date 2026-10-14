"""Count reviewed extraction phenomena; counts do not establish clinical accuracy."""
import argparse
from collections import Counter
import json
from pathlib import Path

from training.teacher_data import read_rows,sha

CORE_FIELDS=('symptoms','red_flags','duration','severity','location','pain_character')


def inspect(path):
    rows=read_rows(path)
    if any(r.get('review_status')!='reviewed' for r in rows):
        raise ValueError('coverage audit needs reviewed data')
    coverage={};gaps=[]
    for split in ('train','validation'):
        samples=[r for r in rows if r['split']==split]
        facts=[f for r in samples for f in r['target']['facts']]
        coverage[split]=dict(rows=len(samples),families=len({r['source_family_id'] for r in samples}),
            fields=dict(Counter(f['field'] for f in facts)),
            polarity=dict(Counter(f['polarity'] for f in facts)),
            temporality=dict(Counter(f['temporality'] for f in facts)),
            subject=dict(Counter(f['subject'] for f in facts)),
            operation=dict(Counter(f['operation'] for f in facts)),
            primary_strata=dict(Counter(r['primary_stratum'] for r in samples)))
        for field in CORE_FIELDS:
            if not coverage[split]['fields'].get(field):gaps.append(dict(split=split,axis='fields',value=field))
        for time in ('current','historical','resolved','hypothetical'):
            if not coverage[split]['temporality'].get(time):gaps.append(dict(split=split,axis='temporality',value=time))
    return dict(scope='reviewed_coverage_diagnostic_not_training_release',source_sha256=sha(path),
        coverage=coverage,core_coverage_gaps=gaps,core_count_coverage_complete=not gaps,
        training_release_allowed=False,
        note='Counts measure presence, not diversity or semantic quality. This gate covers symptoms, flags and attributes; profile/history and other reserved schema fields are not validated by the pilot.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=inspect(a.data)
    with a.out.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({'gaps':result['core_coverage_gaps'],'training_release_allowed':False}))
