"""Opt-in V16: explicit old-fact resolution preserves identity, not an alias."""
import re
from llm.fact_contract_runtime import position_delta
from llm.runtime_extraction_guard import CLINICAL,SYMPTOM_ALIASES,RED_FLAG_SYMPTOMS,check_denial_and_duration
from llm.compact_fact_guard_v15 import pure_control,check_conditions,_remaining_transactions,_additional_checks
from llm.deferred_scope import evidence_scope

VERSION='compact_transaction_guard_v16'


def linked_resolution_allowed(inp,item):
    if not (item['operation']=='resolve' and item['field']=='symptoms' and item['value']=='眩晕'
            and item['subject']=='patient' and item['polarity']=='present' and item['temporality']=='resolved'):
        return False
    old=next((k for k in inp['known_facts'] if k['id']==item['replaces']),None)
    if not old or not (old['status']=='active' and old['field']=='symptoms' and old['value']=='眩晕'
            and old['subject']=='patient' and old['polarity']=='present' and old['temporality']=='current'):
        return False
    candidates=[k for k in inp['known_facts'] if k['field']=='symptoms' and k['value'] in {'头晕','头晕目眩','眩晕'}
                and k['subject']=='patient' and k['temporality']=='current' and k['status'] in {'active','disputed'}
                and k['polarity']=='present']
    if len(candidates)!=1:return False
    scope=evidence_scope(inp['current_turn'],item['evidence'])
    if re.search(r'如果|假设|要是|以后|引用|转述|医生说|父亲|母亲|朋友|[“”「」\"]|不确定|可能|也许|未消失|没好|没有好|没缓解|未缓解',scope):
        return False
    return bool(re.search(r'之前|那个|那阵|毛病|症状',item['evidence']) and
                re.search(r'已经好了|现在(?:已经)?好了|已经消失了|已缓解|现在缓解了',item['evidence']))


def check_supported_rewrite(inp,delta,classification,*,modern=False):
    check_denial_and_duration(inp,delta,classification,modern=modern)
    for item in delta['facts']:
        if item['field']=='symptoms' and item['value']=='眩晕' and '头晕' in item['evidence'] and '眩晕' not in item['evidence']:
            if not linked_resolution_allowed(inp,item):raise ValueError('unapproved_dizziness_symptom_rewrite')


def check_attribute_transactions(inp, delta, classification, *, modern=False):
    """v7 finite typed-update alarms; no fact repair or label conversion.

    A status description such as 未变 is not a new elapsed duration. For an
    explicit correction with existing bound attributes, require a correction
    transaction instead of accepting an asserted conflicting value. This is
    incomplete and may abstain; unchanged ordinary/legacy behavior is retained.
    """
    check_supported_rewrite(inp,delta,classification,modern=modern)
    from llm.fact_contract import ATTRIBUTES
    from llm.fact_normalization import normalize_facts
    normalized,known,_=normalize_facts(delta,inp)
    for item in normalized['facts']:
        if (item['field'] not in ATTRIBUTES or item['subject']!='patient' or item['temporality']!='current'
                or item['polarity']!='present'):
            continue
        status_values = {'未变','没变','不变','没有变化','时间未变','时间没变','不确定'} | ({'没改'} if modern else set())
        if item['value'].strip() in status_values:
            raise ValueError('attribute_status_is_not_an_attribute_value')
        if (item['operation']=='assert' and item['about'] is not None
                and re.search(r'记错|说错|更正|改正',item['evidence'])):
            previous=[f for f in known['known_facts'] if f['field']==item['field']
                and f['about']==item['about'] and f['subject']=='patient' and f['temporality']=='current'
                and f['status'] in {'active','disputed'} and f['polarity']=='present']
            if any(f['value']!=item['value'] for f in previous):
                raise ValueError('explicit_attribute_correction_requires_correct_operation')


def check_semantic_types(inp, delta, classification, *, modern=False):
    """v9: finite rejection checks, not a semantic extraction/scoring oracle.

    Keep the old prompt, facts, aliases and ledger intact. Unsupported normal
    evidence must not introduce observed abnormal values in either field.
    Clinical names are not elapsed-time or degree values. No missing fact is
    filled and no normal description is turned into a symptom denial.
    """
    check_attribute_transactions(inp,delta,classification,modern=modern)
    from llm.fact_contract import ATTRIBUTES
    for item in delta['facts']:
        if item['field'] in ATTRIBUTES and CLINICAL.fullmatch(item['value'].strip()):
            raise ValueError('problem_name_is_not_an_attribute_value')
        if (item['field'] in {'symptoms','cold_heat'} and item['polarity']=='present'
                and item['subject']=='patient' and item['temporality']=='current'
                and item['operation']=='assert' and item['value'] in {'发热','畏寒','怕冷'}):
            evidence=item['evidence']
            terms={item['value']} | {a for a,v in SYMPTOM_ALIASES.items() if v==item['value']}
            if re.search(r'正常|和平时一样',evidence) and not any(t in evidence for t in terms):
                raise ValueError('normal_state_does_not_support_specific_abnormality')


def check_problem_corrections(inp, delta, classification):
    """v11 finite transaction alarm; never chooses a replacement or a label.

    A current, same-field opposite assertion citing explicit correction wording
    must be reconsidered by the model. Ordinary temporal changes, other people,
    quotation and conditional language are outside this narrow check.
    """
    check_semantic_types(inp, delta, classification, modern=True)
    for item in delta['facts']:
        if item['operation']=='correct' and item['polarity']=='present':
            old=next((f for f in inp['known_facts'] if f['id']==item['replaces']),None)
            if old and old['polarity']=='negated' and old['status']=='active':
                from memory.fact_temporal import prior_denials_ended
                start=inp['current_turn'].find(item['evidence'])
                if start>=0 and inp['current_turn'].find(item['evidence'],start+1)<0:
                    simulated=[{**f,'start':inp['current_turn'].find(f['evidence'])} for f in delta['facts']]
                    for f in simulated:
                        if f['id']==item['id']:f['operation']='assert'
                    ended=prior_denials_ended(inp,simulated,inp['known_facts'],
                        {f['id']:f['status'] for f in inp['known_facts']})
                    if any(t['prior_id']==old['id'] for t in ended):
                        raise ValueError('explicit_new_occurrence_must_not_correct_prior_denial')
        if (item['operation'] != 'assert' or item['field'] not in {'symptoms','red_flags','sleep'}
                or item['subject'] != 'patient' or item['temporality'] != 'current'
                or item['polarity'] not in {'present','negated'}):
            continue
        evidence = item['evidence']
        if not re.search(r'说错|记错|更正|改正', evidence):
            continue
        if re.search(r'如果|假设|要是|引用|[“”「」\"]', evidence):
            continue
        previous = [f for f in inp['known_facts'] if
            all(f[k] == item[k] for k in ('field','value','subject','temporality','about'))
            and f['status'] in {'active','disputed'} and f['polarity'] in {'present','negated'}]
        if any(f['polarity'] != item['polarity'] for f in previous):
            raise ValueError('explicit_problem_correction_requires_correct_operation')


def check(inp,delta,classification):
    position_delta(delta,inp)
    effective=dict(classification)
    if pure_control(inp['current_turn']):effective['class']='possible_control'
    check_problem_corrections(inp,delta,effective)
    check_conditions(inp,delta)
    _remaining_transactions(inp,delta,effective)
    _additional_checks(inp,delta,effective)
