"""Finite rejection alarms for v5; never manufactures an extracted fact."""
import re
from llm.deferred_scope import evidence_scope
from llm.fact_contract_runtime import position_delta
from llm.runtime_extraction_guard import check_conditional_coverage

VERSION='semantic_transaction_guard_v13'
CALENDAR=re.compile(r'^(?:上个月|下个月|本月|这个月|去年|今年|明年|昨天|今天|明天|前天)(?:上午|下午|晚上)?$')
RESOLVED=re.compile(r'(?:现在|目前|已经|已)(?:明显)?(?:缓解|好转|好了)')
BLOCK=re.compile(r'如果|假设|要是|引用|转述|[“”「」"]|并非|不是|未缓解|没有缓解|没缓解')


def check_transactions(inp,delta,classification):
    position_delta(delta,inp)  # Includes subject/field/binding/replacement checks.
    check_conditional_coverage(inp,delta,classification)
    facts=delta['facts']
    for f in facts:
        old=next((k for k in inp['known_facts'] if k['id']==f['replaces']),None)
        if f['operation']=='correct' and f['polarity']=='present' and old and old['polarity']=='negated':
            from memory.fact_temporal import prior_denials_ended
            from memory.fact_identity import VERSION as IDENTITY
            from llm.fact_projection_policy import BREATHING_WORDS
            from llm.fact_lexical_policy import PAIN_ALIASES,APPROVED_SYMPTOM_ALIASES
            simulated=[dict(g,start=inp['current_turn'].find(g['evidence']),
                operation='assert' if g['id']==f['id'] else g['operation']) for g in facts]
            transitions=prior_denials_ended(inp,simulated,inp['known_facts'],
                {k['id']:k['status'] for k in inp['known_facts']},identity_version=IDENTITY,
                lexical_aliases={**PAIN_ALIASES,**APPROVED_SYMPTOM_ALIASES,**BREATHING_WORDS})
            if any(t['prior_id']==old['id'] for t in transitions):
                raise ValueError('explicit_new_occurrence_must_not_correct_prior_denial')
        if f['field']=='duration' and CALENDAR.fullmatch(f['value'].strip()):
            raise ValueError('calendar_reference_is_not_elapsed_duration')
        if not (f['field'] in {'symptoms','red_flags'} and f['subject']=='patient'
                and f['temporality']=='current' and f['polarity']=='present'):
            continue
        scope=evidence_scope(inp['current_turn'],f['evidence'])
        if BLOCK.search(scope):continue
        if re.search(r'(?:疼|痛)得厉害',scope):
            degree=[g for g in facts if g['field']=='severity' and g['about']==f['id']
                    and (g['value'],g['polarity'],g['subject'],g['temporality'])==(
                        '厉害','present','patient','current')]
            known_obj=[k for k in inp['known_facts'] if k['status']=='active'
                and all(k[key]==f[key] for key in ('field','value','polarity','subject','temporality'))]
            degree+= [k for k in inp['known_facts'] if k['field']=='severity' and k['value']=='厉害'
                and k['polarity']=='present' and k['status']=='active'
                and any(k['about']==obj['id'] for obj in known_obj)]
            if not degree:raise ValueError('explicit_degree_has_no_bound_attribute')
        if f['operation']=='assert' and RESOLVED.search(scope):
            if not any(g['operation']=='resolve' and g['value']==f['value']
                       and g['subject']==f['subject'] for g in facts):
                raise ValueError('explicit_resolution_requires_resolved_operation')
