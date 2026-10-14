"""Fact-card teacher pilot with immutable proposals; defaults to zero-call dry run."""
import argparse
from dataclasses import asdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path

from llm.clients import ModelClient, RunBudget, load_role_configs
from llm.extraction import FIELDS, PROMPT, build_input, messages, validate_delta

TEACHER_PROMPT = '''你为信息抽取构造虚构工程训练候选，不是真实患者，也不提供医疗建议。
根据事实卡用自然口语表达恰好指定的 facts；可以改变表达，但不得增删事实、加入新症状、修改极性或更正关系。
known_state 是背景，不能当成当前轮已重述的信息。style 只控制措辞，不增添事实。
返回 JSON 对象，且只包含 current_turn 字符串和 target 对象。target 的格式遵循以下抽取契约。
不要输出思维过程、解释、风险、诊断或处置。教师结果只是候选，需要后续审核。
''' + PROMPT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def validate_cards(cards):
    seen, families = set(), {}
    for card in cards:
        if not isinstance(card.get('sample_id'), str) or not card['sample_id'] or card['sample_id'] in seen:
            raise ValueError('duplicate/invalid sample_id')
        seen.add(card['sample_id'])
        family, split = card.get('family_id'), card.get('split')
        if not isinstance(family, str) or not family or split not in {'train', 'validation'}:
            raise ValueError('invalid family/split')
        if family in families and families[family] != split:
            raise ValueError('family leakage across train/validation')
        families[family] = split
        build_input('占位', card['turn_id'], card['known_state'])
        facts = card['facts']
        if not isinstance(facts, list) or len(facts) > 12:
            raise ValueError('facts must be a short list')
        for fact in facts:
            if set(fact) != {'field', 'value', 'polarity', 'operation'} or fact['field'] not in FIELDS:
                raise ValueError('invalid fact fields')
            if not isinstance(fact['value'], str) or not fact['value'] or len(fact['value']) > 120:
                raise ValueError('invalid fact value')
            if fact['polarity'] not in {'present', 'negated', 'uncertain'} or fact['operation'] not in {'add', 'correct', 'deny'}:
                raise ValueError('invalid fact polarity/operation')
            if (fact['operation'] == 'deny') != (fact['polarity'] == 'negated'):
                raise ValueError('inconsistent denial fact')
        keys = [tuple(f[k] for k in ('field', 'value', 'polarity', 'operation')) for f in facts]
        if len(keys) != len(set(keys)):
            raise ValueError('duplicate fact')
    if not cards:
        raise ValueError('empty cards')


def request_messages(card, system_prompt=TEACHER_PROMPT):
    return [{'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': json.dumps({k: card[k] for k in ('turn_id', 'known_state', 'facts', 'style')}, ensure_ascii=False)}]


def protected_inputs(paths):
    # Only identities and input text enter the overlap checker; no evaluation labels go to Teacher.
    result = []
    for path in paths:
        for row in read_rows(path):
            if 'turns' in row:
                result.extend({'sample_id': row['scenario_id'], 'family_id': row['family_id'],
                               'text': turn['user']} for turn in row['turns'])
            elif 'input' in row and 'current_turn' in row['input']:
                result.append({'sample_id': row['sample_id'], 'family_id': row['family_id'],
                               'text': row['input']['current_turn']})
            else:
                raise ValueError(f'unsupported protected input shape: {path}')
    return result


def check_candidate(card, proposal, protected=()):
    errors, overlaps, candidate = [], [], None
    try:
        if not isinstance(proposal, dict) or set(proposal) != {'current_turn', 'target'}:
            raise ValueError('teacher envelope requires exactly current_turn and target')
        inp = build_input(proposal['current_turn'], card['turn_id'], card['known_state'])
        target = validate_delta(proposal['target'], inp)
        keys = ('field', 'value', 'polarity', 'operation')
        actual = sorted(tuple(m[k] for k in keys) for m in target['mentions'])
        expected = sorted(tuple(m[k] for k in keys) for m in card['facts'])
        if actual != expected:
            errors.append('target_facts_differ_from_authored_card')
        for prior in protected:
            a, b = ''.join(inp['current_turn'].split()), ''.join(prior['text'].split())
            score = SequenceMatcher(None, a, b).ratio()
            if card['sample_id'] == prior['sample_id'] or card['family_id'] == prior['family_id'] or score >= .8:
                overlaps.append({'sample_id': prior['sample_id'], 'similarity': score})
        if overlaps:
            errors.append('possible_input_or_family_overlap')
        candidate = {'sample_id': card['sample_id'], 'family_id': card['family_id'], 'split': card['split'],
                     'input': inp, 'target': target, 'messages': messages(inp, target),
                     'origin': 'teacher_synthetic_candidate', 'review_status': 'draft',
                     'eligible_for_training': False}
    except (ValueError, KeyError, TypeError) as exc:
        errors.append(str(exc))
    return {'candidate': candidate, 'program_issues': errors, 'overlaps': overlaps,
            'mechanical_pass': not errors, 'semantic_review': 'pending',
            'note': 'Card agreement and valid spans do not prove semantic correctness.'}


def run(cards, out, config_path, protected_paths, *, execute=False, limit_cny=10.0, system_prompt=TEACHER_PROMPT):
    validate_cards(cards)
    if not 0 < limit_cny <= 100:
        raise ValueError('per-run budget must be in (0,100] CNY')
    configs = load_role_configs(config_path)
    cfg = configs['teacher']
    protected = protected_inputs(protected_paths)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running' if execute else 'dry_run', 'teacher': cfg.public_summary(),
                'cards': len(cards), 'calls': 0, 'mechanical_pass': 0, 'human_approved': 0,
                'budget_limit_cny': limit_cny, 'estimated_accounted_cny': 0,
                'prompt_sha256': hashlib.sha256(system_prompt.encode()).hexdigest(),
                'protected_inputs_sha256': {str(p): sha(p) for p in protected_paths},
                'source_sha256': {p: sha(p) for p in ('training/teacher_data.py', 'llm/extraction.py', 'llm/clients.py')},
                'intended_use': 'User-confirmed personal task-specific extraction SFT; not a competing product. No vendor approval claimed.', 'no_automatic_training_export': True}
    for card in cards:
        append(out / 'fact_cards.jsonl', card)
        append(out / 'requests.jsonl', {'sample_id': card['sample_id'], 'messages': request_messages(card, system_prompt)})
    manifest['fact_cards_sha256'] = sha(out / 'fact_cards.jsonl')
    manifest['requests_sha256'] = sha(out / 'requests.jsonl')
    write_json(out / 'manifest.json', manifest)
    if not execute:
        return manifest
    budget = RunBudget(limit_cny)
    client = ModelClient(cfg, budget=budget)
    # Failed calls and rejected labels remain in raw_results. No hidden retry or resume replay.
    generated = []
    try:
        for card in cards:
            result = client.call(request_messages(card, system_prompt), require_json=True)
            manifest['calls'] += 1
            append(out / 'raw_results.jsonl', {'sample_id': card['sample_id'], 'result': asdict(result)})
            try:
                proposal = json.loads(result.content) if result.ok else None
                checks = check_candidate(card, proposal, protected + generated)
            except (ValueError, TypeError) as exc:
                proposal, checks = None, {'mechanical_pass': False, 'program_issues': [str(exc)]}
            append(out / 'proposals.jsonl', {'sample_id': card['sample_id'], 'original_proposal': proposal, 'checks': checks})
            manifest['mechanical_pass'] += checks['mechanical_pass']
            manifest['estimated_accounted_cny'] = budget.accounted_cny
            if isinstance(proposal, dict) and isinstance(proposal.get('current_turn'), str):
                generated.append({'sample_id': card['sample_id'], 'family_id': '', 'text': proposal['current_turn']})
            write_json(out / 'manifest.json', manifest)
            if not result.ok:
                manifest['status'] = 'stopped:' + result.status
                break
        else:
            manifest['status'] = 'awaiting_semantic_review'
    except BaseException:
        manifest['status'] = 'interrupted'
        raise
    finally:
        manifest['estimated_accounted_cny'] = budget.accounted_cny
        write_json(out / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cards', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('training/teacher_beijing_flash_v1.yaml'))
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--limit-cny', type=float, default=10)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--system-prompt', type=Path)
    args = parser.parse_args()
    if not 1 <= args.limit <= 1000:
        parser.error('limit must be 1..1000')
    cards = read_rows(args.cards)
    validate_cards(cards)  # Also check split leakage outside selected prefix.
    protected = [Path('evaluation/datasets') / name for name in
                 ('dev_reviewed_v1.jsonl', 'frozen_v1.jsonl', 'strategy_challenge_draft_v1.jsonl')]
    result = run(cards[:args.limit], args.out, args.config, protected,
                 execute=args.execute, limit_cny=args.limit_cny,
                 system_prompt=args.system_prompt.read_text(encoding="utf-8-sig") if args.system_prompt else TEACHER_PROMPT)
    print(json.dumps({k: result[k] for k in ('status', 'cards', 'calls', 'mechanical_pass', 'human_approved')}))


if __name__ == '__main__':
    main()
