"""Request permissions derived from verified records; raw text grants none."""
from copy import deepcopy
import hashlib
import json
import re
from agent.planner import PLANNER_CONFIG
from agent.qwen_interaction_policy import PLAN_SCHEMA as LEGACY_SCHEMA,paused,request_types

VERSION='request_authority_v1'
KINDS=['verified_request','safety','stage_summary','default_collection','clarification','persistent_preference']
STATE_REFS=['risk_result.risk','missing_slots','stage_summary.eligible','interaction_turn.pending',
    'interaction_preferences.followup_paused','case_state.contradictions','fact_update_failed','should_request_pulse']
PLAN_SCHEMA=deepcopy(LEGACY_SCHEMA)
PLAN_SCHEMA['required']+=['basis','request_ids','state_refs']
PLAN_SCHEMA['properties'].update(basis={'enum':KINDS},
    request_ids={'type':'array','maxItems':8,'uniqueItems':True,'items':{'type':'string','maxLength':40}},
    state_refs={'type':'array','maxItems':4,'uniqueItems':True,'items':{'enum':STATE_REFS}})


def requests(state):
    turn=state.get('interaction_turn',{});result=[]
    for record in turn.get('requests',[]):
        identity=[turn.get('text',''),state.get('interaction_sequence',0),record['type'],record['scope']]
        rid='req_'+hashlib.sha256(json.dumps(identity,ensure_ascii=False).encode()).hexdigest()[:12]
        result.append({**deepcopy(record),'id':rid})
    return result


def stage_summary(state,risk,base):
    total=sum(state.get('followup_counts',{}).values())
    eligible=(risk.get('risk')!='HIGH' and not paused(state) and not state.get('contradictions')
        and not state.get('fact_extraction_error') and not state.get('interaction_turn',{}).get('pending')
        and total>=PLANNER_CONFIG['summarize_total_followups']
        and len(base.get('missing_slots',[]))>=PLANNER_CONFIG['summarize_min_missing'])
    return {'eligible':bool(eligible),'total_followups':total,'missing_count':len(base.get('missing_slots',[])),
        'minimum_followups':PLANNER_CONFIG['summarize_total_followups'],'minimum_missing':PLANNER_CONFIG['summarize_min_missing'],
        'meaning':'stage milestone; does not establish that the patient made no progress'}


def bind(plan,state,risk,base):
    plan=deepcopy(plan);records=requests(state);kinds=request_types(state);action=plan['next_action']
    plan.update(basis='default_collection',request_ids=[],state_refs=['missing_slots'])
    selected=set()
    if action=='risk_escalation':plan.update(basis='safety',state_refs=['risk_result.risk'])
    elif action=='summarize_progress':
        selected=kinds & {'summarize','request_references'}
        if selected:plan.update(basis='verified_request',state_refs=[])
        else:plan.update(basis='stage_summary',state_refs=['stage_summary.eligible'])
    elif action=='provide_hospital_options':
        selected={'find_hospitals'}|(kinds&{'request_references'});plan.update(basis='verified_request',state_refs=[])
    elif action=='acknowledge_pause':
        if 'pause_followup' in kinds:selected={'pause_followup'};plan.update(basis='verified_request',state_refs=[])
        else:plan.update(basis='persistent_preference',state_refs=['interaction_preferences.followup_paused'])
    elif action=='clarify_interaction_request':
        plan.update(basis='clarification',state_refs=[])
        if state.get('interaction_turn',{}).get('pending'):plan['state_refs']=['interaction_turn.pending']
        elif state.get('fact_extraction_error'):plan['state_refs']=['fact_update_failed']
        elif 'find_hospitals' in kinds:selected={'find_hospitals'}
    elif action=='clarify_conflict':plan.update(basis='clarification',state_refs=['case_state.contradictions'])
    elif action=='request_pulse_input':plan['state_refs']=['should_request_pulse']
    plan['request_ids']=[r['id'] for r in records if r['type'] in selected]
    return plan


def validate_basis(plan,state,risk,base):
    records={r['id']:r for r in requests(state)};ids=plan['request_ids'];refs=plan['state_refs'];basis=plan['basis'];action=plan['next_action']
    if any(i not in records for i in ids):raise ValueError('unknown_current_request_id')
    cited={records[i]['type'] for i in ids}
    allowed_refs={'verified_request':set(),'safety':{'risk_result.risk'},'stage_summary':{'stage_summary.eligible'},
        'default_collection':{'missing_slots','should_request_pulse'},'persistent_preference':{'interaction_preferences.followup_paused'},
        'clarification':{'interaction_turn.pending','fact_update_failed','case_state.contradictions'}}
    if any(ref not in allowed_refs[basis] for ref in refs):raise ValueError('state_ref_does_not_support_basis')
    active_refs={'interaction_turn.pending':bool(state.get('interaction_turn',{}).get('pending')),
        'fact_update_failed':bool(state.get('fact_extraction_error')),'case_state.contradictions':bool(state.get('contradictions'))}
    if any(ref in active_refs and not active_refs[ref] for ref in refs):raise ValueError('referenced_state_is_not_active')
    if basis!='verified_request' and ids and not (basis=='clarification' and cited=={'find_hospitals'}):raise ValueError('request_id_with_wrong_basis')
    if basis=='verified_request' and not ids:raise ValueError('verified_basis_without_request_id')
    if action=='summarize_progress':
        if basis=='verified_request':
            if not cited & {'summarize','request_references'}:raise ValueError('request_does_not_authorize_summary')
        elif basis=='stage_summary':
            if not stage_summary(state,risk,base)['eligible'] or 'stage_summary.eligible' not in refs:raise ValueError('stage_summary_not_eligible')
        else:raise ValueError('summary_requires_request_or_stage_basis')
    elif action=='provide_hospital_options':
        if basis!='verified_request' or 'find_hospitals' not in cited:raise ValueError('hospital_request_basis_missing')
    elif action=='acknowledge_pause':
        if basis=='verified_request':
            if 'pause_followup' not in cited:raise ValueError('decline_pulse_is_not_pause')
        elif basis=='persistent_preference':
            if not state.get('interaction_preferences',{}).get('followup_paused') or 'interaction_preferences.followup_paused' not in refs:raise ValueError('inactive_pause_preference')
        else:raise ValueError('pause_basis_missing')
    elif action=='risk_escalation':
        if basis!='safety' or risk.get('risk')!='HIGH' or 'risk_result.risk' not in refs:raise ValueError('safety_basis_missing')
    elif action in {'clarify_conflict','clarify_interaction_request'}:
        if basis!='clarification':raise ValueError('clarification_basis_required')
        valid=(('interaction_turn.pending' in refs and bool(state.get('interaction_turn',{}).get('pending')))
            or ('fact_update_failed' in refs and bool(state.get('fact_extraction_error')))
            or ('case_state.contradictions' in refs and bool(state.get('contradictions')))
            or cited=={'find_hospitals'})
        if not valid:raise ValueError('clarification_has_no_state_evidence')
    else:
        if basis!='default_collection':raise ValueError('ordinary_action_uses_default_collection')
        required='should_request_pulse' if action=='request_pulse_input' else 'missing_slots'
        if required not in refs:raise ValueError('collection_state_ref_missing')
    if 'retrieval' in plan['tool_requests'] and 'request_references' not in cited:raise ValueError('retrieval_request_not_cited')
    if 'hospital' in plan['tool_requests'] and 'find_hospitals' not in cited:raise ValueError('hospital_request_not_cited')
    return True


def reason_alarms(reason,state):
    """Finite observed false-claim alarms, not a general free-text proof checker."""
    alarms=[];turn=state.get('interaction_turn',{});kinds=request_types(state)
    def asserted(pattern):
        for match in re.finditer(pattern,reason):
            before=reason[max(0,match.start()-8):match.start()];after=reason[match.end():match.end()+4]
            if re.search(r'如果|假如|若|不能|并非|不是|不代表|并不',before) or re.match(r'时|的话',after):continue
            return True
        return False
    if asserted(r'requests\s*(?:为|是)?空|请求列表为空') and turn.get('requests'):alarms.append('reason_falsely_claims_requests_empty')
    if asserted(r'pending\s*(?:非空|不为空)') and not turn.get('pending'):alarms.append('reason_falsely_claims_pending_nonempty')
    if asserted(r'(?:存在|已收到|患者提出)(?:了)?资料请求') and 'request_references' not in kinds:alarms.append('reason_falsely_claims_reference_request')
    return alarms
