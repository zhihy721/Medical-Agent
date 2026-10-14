"""Export a fixed fictional extractor workload, without supervised answers."""
import argparse
import json
from pathlib import Path

from llm.fact_contract import messages
from training.teacher_data import read_rows, sha


def build(source, out):
    source, out = Path(source), Path(out)
    rows = read_rows(source)
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('empty or duplicate source')
    workload=[]
    for row in rows:
        if row.get('review_status') != 'reviewed':
            raise ValueError('unreviewed workload source')
        prompt=messages(row['input'])
        if row.get('messages') != messages(row['input'],row['target']):
            raise ValueError('source has stale prompt/target serialization')
        workload.append(dict(sample_id=row['sample_id'],messages=prompt))
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as stream:
        for row in workload:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    manifest=dict(scope='fictional_infrastructure_workload_not_quality_test',
                  source_sha256=sha(source),workload_sha256=sha(out),records=len(workload),
                  prompt_only=True,training_answers_excluded=True,
                  source_splits=sorted({r['split'] for r in rows}))
    with out.with_suffix('.manifest.json').open('x',encoding='utf-8') as stream:
        json.dump(manifest,stream,ensure_ascii=False,indent=2)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.source,args.out),ensure_ascii=False))


if __name__=='__main__':
    main()
