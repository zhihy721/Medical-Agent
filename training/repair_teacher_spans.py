"""Derive unique literal evidence offsets without changing immutable Teacher output."""
import argparse
from copy import deepcopy
from pathlib import Path
from training.teacher_data import read_rows, write_json, append, sha, check_candidate, protected_inputs


def propose_span_repairs(proposal):
    candidate = deepcopy(proposal)
    changes, issues = [], []
    text = candidate.get('current_turn') if isinstance(candidate, dict) else None
    target = candidate.get('target') if isinstance(candidate, dict) else None
    if not isinstance(text, str) or not isinstance(target, dict) or not isinstance(target.get('mentions'), list):
        return candidate, changes, ['invalid_teacher_envelope']
    for index, mention in enumerate(target['mentions']):
        if not isinstance(mention, dict):
            issues.append(f'mention_{index}:invalid');continue
        evidence = mention.get('evidence')
        if not isinstance(evidence, str) or not evidence:
            issues.append(f'mention_{index}:missing_evidence');continue
        start, end = mention.get('start'), mention.get('end')
        if type(start) is int and type(end) is int and 0 <= start < end <= len(text) and text[start:end] == evidence:
            continue
        # Count overlapping matches too: str.count is insufficient for strings such as 哈哈哈.
        positions = [i for i in range(len(text)) if text.startswith(evidence, i)]
        if len(positions) != 1:
            issues.append(f'mention_{index}:evidence_not_unique_or_absent');continue
        new_start = positions[0];new_end = new_start + len(evidence)
        changes.append({'mention_index':index,'before':{'start':start,'end':end},
                        'after':{'start':new_start,'end':new_end},'reason':'unique_literal_substring'})
        mention.update(start=new_start,end=new_end)
    return candidate, changes, issues


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    cards={r['sample_id']:r for r in read_rows(args.run/'fact_cards.jsonl')}
    proposals=read_rows(args.run/'proposals.jsonl')
    source_hashes={name:sha(args.run/name) for name in ('fact_cards.jsonl','proposals.jsonl','raw_results.jsonl','manifest.json')}
    paths=[Path('evaluation/datasets')/name for name in ('dev_reviewed_v1.jsonl','frozen_v1.jsonl','strategy_challenge_draft_v1.jsonl')]
    protected=protected_inputs(paths)
    seen=[];total_changed=total_pass=0
    args.out.mkdir(parents=True,exist_ok=False)
    for row in proposals:
        fixed,changes,issues=propose_span_repairs(row['original_proposal'])
        card=cards[row['sample_id']]
        checks=check_candidate(card,fixed,protected+seen)
        total_changed+=bool(changes);total_pass+=checks['mechanical_pass'] and not issues
        append(args.out/'repair_proposals.jsonl',{'sample_id':row['sample_id'],'original_proposal':row['original_proposal'],
                'original_program_checks':row['checks'],'proposed_after_span_repair':fixed,'changes':changes,
                'repair_issues':issues,'post_repair_checks':checks,'review_status':'draft','eligible_for_training':False})
        if isinstance(fixed,dict) and isinstance(fixed.get('current_turn'),str):
            seen.append({'sample_id':card['sample_id'],'family_id':'','text':fixed['current_turn']})
    if source_hashes!={name:sha(args.run/name) for name in source_hashes}:
        raise RuntimeError('source changed during repair proposal generation')
    summary={'rows':len(proposals),'rows_with_span_proposals':total_changed,'mechanical_pass_after_proposals':total_pass,
             'human_approved':0,'teacher_api_calls':0,'source_hashes':source_hashes,
             'protected_hashes':{str(p):sha(p) for p in paths},'code_sha256':sha(__file__),
             'note':'Derived suggestions only; original output unchanged. No semantic corrections or training approval.'}
    write_json(args.out/'summary.json',summary)
    print(summary)


if __name__=='__main__':main()
