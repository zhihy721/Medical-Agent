"""Finite, audited serialization conversions; never supplies missing facts."""
from copy import deepcopy
import re

from llm.fact_normalization import canonical_value

VERSION = 'fact_representation_v3'
SLEEP = frozenset({'失眠', '多梦', '早醒', '入睡困难', '睡觉多梦'})
DEGREES = {'很轻': '轻度', '偏轻': '轻度', '轻微': '轻度', '轻': '轻度',
           '中等': '中度', '很重': '重度', '剧烈': '重度'}
NEGATIVE_VALUES = {'不失眠': ('sleep', '失眠'), '没有失眠': ('sleep', '失眠'),
                   '无失眠': ('sleep', '失眠'), '没有慢性病': ('past_history', '慢性病')}


def literal_evidence(text, evidence):
    """Only whitespace variation, with one occurrence and original offsets.

    No punctuation removal, synonym replacement, case-folding or fuzzy match.
    Multiple matches remain invalid. Recovered evidence is a literal substring.
    """
    if evidence in text:
        return evidence
    needle = ''.join(c for c in evidence if not c.isspace())
    chars = [(i, c) for i, c in enumerate(text) if not c.isspace()]
    compact = ''.join(c for _, c in chars)
    if not needle:
        return evidence
    matches = [m.start() for m in re.finditer('(?=' + re.escape(needle) + ')', compact)]
    if len(matches) != 1:
        return evidence
    start = chars[matches[0]][0]
    end = chars[matches[0] + len(needle) - 1][0] + 1
    return text[start:end]


def normalize_representation(delta, inp, *, alias_version=None):
    from llm.fact_normalization import VERSION as ALIAS_VERSION
    from llm.fact_lexical_policy import ALIAS_VERSION as NEW_ALIAS_VERSION
    from llm.fact_projection_policy import ALIAS_VERSION as V5
    alias_version = alias_version or ALIAS_VERSION
    canonical_value('', '', version=alias_version)
    conversion_version = ('fact_representation_v5' if alias_version==V5 else
                          'fact_representation_v4' if alias_version == NEW_ALIAS_VERSION else VERSION)
    output = deepcopy(delta)
    changes = []
    if alias_version in {NEW_ALIAS_VERSION,V5}:
        if alias_version==V5:
            from llm.fact_projection_policy import align
        else:
            from llm.fact_policy_alignment import align
        output, _, field_changes = align(output, inp)
        changes.extend(dict(change, kind='finite_field_alignment', representation_version=conversion_version)
                       for change in field_changes)
    for f in output['facts']:
        before = deepcopy(f)
        # Conversion is confined to facts already proposed by the model.
        if f['field'] in {'symptoms', 'sleep'} and f['value'] in SLEEP:
            f['field'] = 'sleep'
            if f['value'] == '睡觉多梦':
                f['value'] = '多梦'
        if f['field'] == 'cold_heat' and f['value'] in {'发热', '发烧', '畏寒'}:
            f['field'] = 'symptoms'
        if f['field'] == 'severity':
            f['value'] = DEGREES.get(f['value'], f['value'])
        f['value'] = canonical_value(f['field'], f['value'], version=alias_version)
        negative = NEGATIVE_VALUES.get(f['value'])
        if (negative and f['polarity'] == 'present' and f['field'] in {negative[0], 'symptoms'}
                and f['value'] in f['evidence']
                and not re.search(r'不是|并非|不能|不确定|不知道|说不准', f['evidence'])):
            f.update(field=negative[0], value=negative[1], polarity='negated')
        f['evidence'] = literal_evidence(inp['current_turn'], f['evidence'])
        if before != f:
            changes.append(dict(kind='representation', fact_id=f['id'], before=before,
                                after=deepcopy(f), version=conversion_version))
    # An uncertain assertion solely deferring a known positive must not create
    # a dispute and thereby hide the actual symptom. Keep the raw proposal in
    # the audit; this is retention of existing evidence, not new extraction.
    from llm.deferred_scope import is_deferred_ambiguous, explicitly_current, evidence_scope
    retained = []
    for f in output['facts']:
        known = next((k for k in inp['known_facts'] if k['field']==f['field'] and k['value']==f['value']
                      and k['subject']==f['subject'] and k['temporality']==f['temporality']
                      and k['polarity']=='present' and k['status']=='active'), None)
        if (known and f['operation']=='assert' and f['polarity']=='uncertain'
                and f['field'] in {'symptoms','red_flags','sleep','sweating','appetite','thirst','cold_heat','emotion','stool_urine'}
                and is_deferred_ambiguous(inp['current_turn'], f['evidence'],modern=True)
                and not explicitly_current(f, evidence_scope(inp['current_turn'],f['evidence']))
                and not any(g['about']==f['id'] for g in output['facts'])):
            changes.append(dict(kind='retain_known_deferred', fact_id=f['id'], known_id=known['id'],
                                original=deepcopy(f), version=conversion_version))
        else:
            retained.append(f)
    output['facts'] = retained
    # Exact assert duplicates exposed by field conversion share one identity.
    # Keep aliases in the trace, redirect local attribute references, never merge
    # opposite polarity, different time/subject, or replacement transactions.
    kept, roots, aliases = [], {}, {}
    for f in output['facts']:
        key = tuple(f[k] for k in ('field', 'value', 'polarity', 'subject', 'temporality', 'about'))
        evidence = f['evidence']
        start = inp['current_turn'].find(evidence)
        valid_evidence = start >= 0 and inp['current_turn'].find(evidence, start+1) < 0
        if f['operation'] == 'assert' and f['replaces'] is None and key in roots and valid_evidence:
            aliases[f['id']] = roots[key]
            changes.append(dict(kind='duplicate_alias', fact_id=f['id'], target_id=roots[key], version=conversion_version))
        else:
            kept.append(f)
            if f['operation'] == 'assert' and f['replaces'] is None and valid_evidence:
                roots[key] = f['id']
    for f in kept:
        if f['about'] in aliases:
            f['about'] = aliases[f['about']]
    output['facts'] = kept
    return output, changes
