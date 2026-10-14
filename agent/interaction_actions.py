"""Approved interaction execution. Models never call these during repair."""
from agent.router import ActionResult


def build(memory,plan,risk):
    from agent.runtime_utils import build_requested_progress_action_result,_nearby_hospital_lines
    state=memory.get_case_state();action=plan['next_action'];turn=state.get('interaction_turn',{})
    if action=='acknowledge_pause':
        text='已按你的要求暂停普通追问。已有症状记录会保留；你可以主动补充信息。'
        if state.get('interaction_preferences',{}).get('followup_paused'):text+='明确说“继续问”后再恢复。'
    elif action=='clarify_interaction_request':
        if turn.get('pending'):text='本轮的交互要求存在冲突或尚未确认。请说明你希望继续问诊、暂停，还是仅整理记录。'
        elif any(r['type']=='find_hospitals' for r in turn.get('requests',[])):
            text='查询医院需要城市或已授权的位置。'
            if state.get('interaction_preferences',{}).get('followup_paused') or any(r['type']=='pause_followup' for r in turn.get('requests',[])):text+='我会保持暂停，等你主动提供位置。'
            else:text+='请告诉我希望查询的城市；也可以在设置中提供位置。'
        else:text='本轮信息未能可靠更新，已有记录仍保留。连接恢复后你可以继续；本轮不生成新的诊断或结论。'
    elif action=='provide_hospital_options':
        lines=_nearby_hospital_lines(state)
        text='\n'.join(lines) if lines else '本轮未获得可用的医院结果，未生成医院清单。'
        if 'retrieval' in plan.get('tool_requests',[]):
            summary=build_requested_progress_action_result(memory,{**plan,'requested_progress':True,'requested_references':True})
            text+='\n'+summary.response
    else:raise ValueError('unsupported interaction action')
    if state.get('pulse_declined'):text+='\n已记录你拒绝脉诊；不会据此取消必要安全提示。'
    return ActionResult(action,text,'COLLECTING_INFO',False,plan.get('missing_slots',[]),render_mode='final')
