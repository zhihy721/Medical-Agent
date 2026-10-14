"""Finite unsupported-denial alarm, not a complete response fact checker."""
import re

VERSION = 'response_denial_guard_v1'
CONCEPTS = ('呼吸困难', '持续高热')


def unsupported_denials(response, state):
    supported = {f['value'] for f in state.get('fact_view', {}).get('facts', [])
        if f['field'] in {'symptoms','red_flags'} and f['polarity']=='negated'
        and f['subject']=='patient' and f['temporality']=='current'
        and f.get('status','active')=='active'}
    unsupported = []
    for clause in re.split(r'[。，,；;！!？?\n]', response):
        # Hypothetical safety advice is not a claim about present absence.
        if re.search(r'如果|假如|若|一旦|尚未(?:询问|确认|记录)|未(?:询问|确认|记录)', clause):
            continue
        match = re.search(r'(?:没有|不伴有|不伴|否认|无)([^，,:：]{0,32})', clause)
        if match:
            unsupported.extend(c for c in CONCEPTS if c in match.group(1) and c not in supported)
    return sorted(set(unsupported))
