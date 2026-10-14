"""Cross-batch family/semantic audit before generation; not a proof of independence."""
from collections import Counter, defaultdict
import hashlib
import json

from llm.fact_contract import position_delta
from llm.fact_normalization import normalize_facts


def semantic_fingerprint(card):
    delta, inp, _ = normalize_facts(card['target'], card['input'])
    position_delta(delta, inp)
    known = {f['id']: f for f in inp['known_facts']}
    current = {f['id']: f for f in delta['facts']}
    def meaning(f, refs, visited=()):
        key = f['id']
        if key in visited:
            raise ValueError('cyclic fact relation')
        value = '<number_or_duration>' if f['field'] in {'age', 'duration'} else f['value']
        fields = [f['field'], value, f['polarity'], f['subject'], f['temporality']]
        about = f.get('about')
        bound = refs.get(about) or known.get(about)
        fields.append(meaning(bound, refs, (*visited,key)) if bound else None)
        if 'operation' in f:
            fields.append(f['operation'])
            replaced = known.get(f.get('replaces'))
            fields.append(meaning(replaced, known) if replaced else None)
        return fields
    shape = {'known': sorted([meaning(f, known) for f in known.values()], key=str),
             'current': sorted([meaning(f, current) for f in current.values()], key=str)}
    return hashlib.sha256(json.dumps(shape, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def audit(cards):
    """All batches and dev/holdout registry cards must be supplied together."""
    errors, warnings = [], []
    groups = {key: defaultdict(list) for key in ('source_family_id', 'template_group', 'semantic')}
    ids = set()
    coverage = Counter()
    for card in cards:
        sid = card['sample_id']
        if sid in ids: errors.append({'sample_id':sid,'issue':'duplicate_sample_id'})
        ids.add(sid)
        for name in ('source_family_id', 'template_group', 'primary_stratum', 'split'):
            if not isinstance(card.get(name),str) or not card[name].strip():
                raise ValueError(sid + ': missing ' + name)
        if card['split'] not in {'train','validation','development_contract_review','holdout'}:
            raise ValueError('invalid split')
        if card.get('eligible_for_training'):
            raise ValueError('registry cards must not self-approve training eligibility')
        coverage[(card['split'],card['primary_stratum'])] += 1
        for name in ('source_family_id','template_group'):
            groups[name][card[name]].append(card)
        groups['semantic'][semantic_fingerprint(card)].append(card)
    for name, table in groups.items():
        for key, rows in table.items():
            if len({r['split'] for r in rows}) > 1:
                item = {'issue':name + '_cross_split', 'key':key,
                        'sample_ids':[r['sample_id'] for r in rows]}
                (warnings if name == 'semantic' else errors).append(item)
    for key, rows in groups['source_family_id'].items():
        if rows[0]['split'] in {'train','validation'} and len(rows) > 4:
            errors.append({'issue':'family_exceeds_four_rows','source_family_id':key})
    return {'scope':'candidate_registry_audit_not_training_release', 'errors':errors,
            'requires_semantic_overlap_review':warnings,
            'ready_for_generation':not errors and not warnings,
            'coverage': [{'split':s,'primary_stratum':t,'rows':n} for (s,t),n in sorted(coverage.items())],
            'source_families':len(groups['source_family_id']),
            'note':'Fingerprints omit evidence/IDs and collapse age/duration variants. Collisions require review; absence does not prove independent semantic families.'}
