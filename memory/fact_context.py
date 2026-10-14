"""Bounded fact selection; the ledger and archived facts are never deleted."""
from copy import deepcopy
import json


def select_facts(facts, current_focus=None, *, max_facts=32, byte_budget=6500):
    by_id = {f['id']: f for f in facts}
    selected = {}

    def group(fact):
        result = {fact['id']: fact}
        parent = by_id.get(fact.get('about'))
        if parent:
            result[parent['id']] = parent
        return result

    # Current/disputed and resolved red flags take priority over recency. If all
    # required safety context cannot fit, fail rather than quietly drop it.
    mandatory = [f for f in facts if (f['field']=='red_flags' or
                 (f['field']=='symptoms' and f['value']=='呼吸困难')) and f['subject']=='patient'
                 and f['polarity']=='present' and f['temporality'] in {'current','resolved'}]
    if current_focus in by_id:
        mandatory.append(by_id[current_focus])
        mandatory.extend(f for f in facts if f.get('about')==current_focus)
    for f in mandatory:
        selected.update(group(f))

    def fits(rows):
        return len(rows)<=max_facts and len(json.dumps(list(rows.values()),ensure_ascii=False,separators=(',',':')).encode('utf8'))<=byte_budget

    if not fits(selected):
        raise ValueError('required fact context exceeds budget; explicit compaction required')
    for f in reversed(facts):
        candidate = {**selected, **group(f)}
        if fits(candidate):
            selected = candidate
    # Stable original order retains identities and avoids swapping two bindings.
    output = [deepcopy(f) for f in facts if f['id'] in selected]
    return output, dict(version='fact_context_v1', total=len(facts), retained=len(output),
                        omitted_ids=[f['id'] for f in facts if f['id'] not in selected],
                        byte_budget=byte_budget, archive_modified=False)
