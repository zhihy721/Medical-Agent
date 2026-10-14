"""Opt-in presentation projection; never changes model output or risk input."""
from copy import deepcopy

VERSION = 'denied_red_flag_projection_v1'
# Identity mappings only. No chest-pain/persistent-chest-pain or stool mapping.
IDENTICAL_DENIED_CONCEPTS = frozenset({'呼吸困难'})


def derive_identical_concept_fields(facts):
    """Same-concept display/score copies in every scope; never changes risk.

    This is not clinical synonym expansion. Polarity, time, subject, evidence,
    and attribute binding are kept. Existing conflicting facts are not removed.
    """
    output, added = deepcopy(facts), []
    for fact in facts:
        if fact['field'] not in {'symptoms', 'red_flags'} or fact['value'] != '呼吸困难':
            continue
        target = 'red_flags' if fact['field'] == 'symptoms' else 'symptoms'
        if any(f['field'] == target and all(f.get(k) == fact.get(k) for k in
               ('value', 'polarity', 'subject', 'temporality', 'about', 'about_value')) for f in output):
            continue
        copy = deepcopy(fact)
        copy.update(field=target, derived_from=fact.get('id'), projection_version='identical_concept_fields_v2')
        output.append(copy)
        added.append(dict(source_id=fact.get('id'), source_field=fact['field'], target_field=target,
                          value=fact['value'], polarity=fact['polarity'], temporality=fact['temporality']))
    return output, dict(version='identical_concept_fields_v2', added=added, risk_input_changed=False)


def derive_denied_red_flags(raw_view):
    view = deepcopy(raw_view)
    added = []
    for fact in raw_view['facts']:
        if not (fact['field'] == 'symptoms' and fact['value'] in IDENTICAL_DENIED_CONCEPTS
                and fact['polarity'] == 'negated' and fact['subject'] == 'patient'
                and fact['temporality'] == 'current' and fact.get('status', 'active') == 'active'
                and fact.get('derived_from') is None):
            continue
        if any(f['field'] == 'red_flags' and all(f.get(k) == fact.get(k)
                for k in ('value', 'polarity', 'subject', 'temporality', 'about')) for f in view['facts']):
            continue
        derived = deepcopy(fact)
        derived.update(field='red_flags', derived_from=fact['id'], projection_version=VERSION)
        view['facts'].append(derived)
        added.append(dict(source_id=fact['id'], source_field='symptoms', target_field='red_flags',
                          value=fact['value'], polarity='negated'))
    return view, dict(version=VERSION, added=added, risk_input_changed=False)
