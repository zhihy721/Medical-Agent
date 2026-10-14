"""Opt-in v5 exact concept representation; raw words and clinical scope survive."""
from copy import deepcopy

VERSION='semantic_representation_v5'
ALIAS_VERSION='fact_aliases_v5'
BREATHING_WORDS={'喘不上气':'呼吸困难','喘不过气':'呼吸困难'}
SHARED_RISK_VALUES={'呼吸困难','便血'}


def align(delta, inp):
    from llm.fact_policy_alignment import align as previous
    output,context,changes=previous(delta,inp)
    for scope,rows in [('prediction',output['facts']),('known',context['known_facts'])]:
        for f in rows:
            if f['field'] not in {'symptoms','red_flags'}:continue
            before=(f['field'],f['value'])
            f['value']=BREATHING_WORDS.get(f['value'],f['value'])
            if f['value'] in SHARED_RISK_VALUES:f['field']='red_flags'
            if before!=(f['field'],f['value']):
                changes.append(dict(scope=scope,id=f['id'],before=before,
                    after=(f['field'],f['value']),version=VERSION,
                    kind='approved_exact_concept_representation',
                    risk_decided_here=False,scope_and_evidence_preserved=True))
    return output,context,changes
