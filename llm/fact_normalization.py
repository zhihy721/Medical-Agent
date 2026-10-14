"""Versioned, field-scoped lexical aliases; no symptom/risk inference."""
from copy import deepcopy

VERSION = 'fact_aliases_v3'
# Deliberately finite: never globally replace 疼 with 痛, or merge body regions.
SYMPTOM_ALIASES = {
    '头疼': '头痛', '胸疼': '胸痛', '腹疼': '腹痛', '背疼': '背痛',
    '腰疼': '腰痛', '牙疼': '牙痛', '耳疼': '耳痛', '关节疼': '关节痛',
    '耳朵疼': '耳痛', '耳朵痛': '耳痛',
    '背部痛': '背痛', '背部疼': '背痛', '后背疼': '背痛', '后背痛': '背痛',
    '肚子疼': '腹痛', '肚子痛': '腹痛',
    '关节也疼': '关节痛', '关节也痛': '关节痛',
    '没力气': '乏力', '没有力气': '乏力',
    '发烧': '发热',
    '后背疼痛': '背痛', '背部疼痛': '背痛', '耳朵疼痛': '耳痛',
}
DURATION_ALIASES = {'一个礼拜': '一周', '一礼拜': '一周', '一个星期': '一周', '一星期': '一周'}


def canonical_value(field, value, *, version=VERSION):
    from llm.fact_lexical_policy import ALIAS_VERSION, PAIN_ALIASES, APPROVED_SYMPTOM_ALIASES
    from llm.fact_projection_policy import ALIAS_VERSION as V5
    if version not in {VERSION, ALIAS_VERSION, V5}:
        raise ValueError('unsupported normalization version')
    if field == 'symptoms':
        if version in {ALIAS_VERSION,V5}:
            value = PAIN_ALIASES.get(value, value)
            value = APPROVED_SYMPTOM_ALIASES.get(value, value)
        return SYMPTOM_ALIASES.get(value, value)
    if field == 'duration':
        return DURATION_ALIASES.get(value, value)
    return value


def normalize_facts(delta, inp, *, version=VERSION):
    """Return copies and an audit trail. Evidence, IDs, polarity and scope stay exact.

    Validate the returned pair with position_delta before consuming it. In
    particular, aliases may expose duplicates; never silently delete those.
    """
    output, context = deepcopy(delta), deepcopy(inp)
    changes = []
    from llm.fact_lexical_policy import ALIAS_VERSION
    from llm.fact_projection_policy import ALIAS_VERSION as V5
    if version in {ALIAS_VERSION,V5}:
        if version==V5:
            from llm.fact_projection_policy import align
        else:
            from llm.fact_policy_alignment import align
        output, context, field_changes = align(output, context)
        changes.extend(dict(change, kind='finite_field_alignment', normalization_version=version) for change in field_changes)
    for scope, facts in [('current', output['facts']), ('known', context['known_facts'])]:
        for fact in facts:
            value = canonical_value(fact['field'], fact['value'], version=version)
            if value != fact['value']:
                changes.append({'scope': scope, 'id': fact['id'], 'field': fact['field'],
                                'original_value': fact['value'], 'canonical_value': value})
                fact['value'] = value
    return output, context, changes
