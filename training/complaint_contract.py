"""Offline v3 complaint intent contract; not yet wired into Agent memory.

The caller supplies evidence-backed intent annotations. This module never infers
patient priority from symptom order or a keyword regex.
"""
from copy import deepcopy


def build_complaints(text, mentions, confirmed=None):
    positive = {m['value']: m for m in mentions if m['field'] == 'symptoms' and m['polarity'] == 'present'}
    chiefs = deepcopy(confirmed or [])
    seen = set()
    for item in chiefs:
        if set(item) != {'value', 'evidence', 'start', 'end'}:
            raise ValueError('confirmed complaint needs value and intent evidence')
        if item['value'] not in positive or item['value'] in seen:
            raise ValueError('confirmed complaint must be a unique current positive symptom')
        start, end = item['start'], item['end']
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text) or text[start:end] != item['evidence']:
            raise ValueError('invalid intent evidence span')
        seen.add(item['value'])
    return {
        'chief_complaints': chiefs,
        'chief_complaint_candidates': sorted(set(positive) - seen),
        'order_semantics': 'unordered',
    }
