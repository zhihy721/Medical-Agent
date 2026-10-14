'use strict';
const $=id=>document.getElementById(id);
const labels={symptoms:'症状',red_flags:'红旗',duration:'病程',severity:'程度',location:'部位',age:'年龄',sex:'性别',past_history:'既往史',cold_heat:'寒热',sweating:'汗出',thirst:'口渴',appetite:'食欲',sleep:'睡眠',stool_urine:'二便',pain_character:'疼痛性质',emotion:'情绪',complexion:'面色',voice_breath:'声音与气息',odor:'气味',female_cycle:'生理周期'};
const actions={acknowledge_pause:'暂停普通追问',clarify_interaction_request:'确认你的要求',provide_hospital_options:'查询医院信息',risk_escalation:'优先安全分诊',ask_followup_single:'进一步询问一个问题',ask_followup_bundle:'补充几个相关问题',clarify_conflict:'先澄清信息冲突',summarize_progress:'整理当前进展',final_advice:'整理本轮问诊建议',request_pulse_input:'询问是否补充脉诊'};
const nodes={acknowledge_pause:'暂停追问',clarify_interaction_request:'确认交互要求',provide_hospital_options:'医院查询',extract:'事实抽取',risk_assess:'风险检查',plan:'选择下一步',review:'动作检查',respond:'生成回复',ask_followup_single:'单项追问',ask_followup_bundle:'组合追问',risk_escalation:'安全分诊',clarify_conflict:'澄清冲突',final_advice:'最终整理'};
let busy=false, recovering=false, current, sessionId='', pending=null;
const pendingKey='medical-pending-requests-v1';
function savePending(value){pending=value;const map=JSON.parse(localStorage.getItem(pendingKey)||'{}');if(value)map[sessionId]=value;else delete map[sessionId];localStorage.setItem(pendingKey,JSON.stringify(map));$('recover').hidden=!value;}
async function recoverPending(){
  if(!pending||recovering)return;
  recovering=true;
  try{
    const record=await api('/chat/requests/'+encodeURIComponent(pending));
    if(record.status==='completed'){
      savePending(null);$('input').value='';await refresh();notice('该次发送已完成，已恢复会话记录。');
    }else if(['failed','interrupted'].includes(record.status)){
      savePending(null);await refresh();notice('该次发送未完整完成，已保留收到的消息与状态。请检查连接后再发送新的补充。');
    }else notice('该次发送仍在处理，或完成状态尚不明确。请查询状态，不要重复提交。');
  }catch(e){notice('无法确认该次发送的完成状态，未自动重发。恢复连接后请查询发送状态。');}
  recovering=false;setBusy(false);
}
async function api(path,body){const r=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await r.json().catch(()=>({}));if(!r.ok)throw Error(data.message||data.response||`服务暂不可用（${r.status}）`);return data;}
function node(tag,cls,text){const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;}
function notice(text){$('notice').hidden=!text;$('notice').textContent=text||'';}
function appendMessage(role,text){const item=node('article',`message ${role}`);item.append(node('div','message-label',role==='user'?'你':'Medical Agent'),node('div','message-content',text));$('messages').append(item);$('messages').scrollTop=$('messages').scrollHeight;return item;}
async function sendChat(text){
  delete document.body.dataset.safetyReceivedMs;delete document.body.dataset.finalReceivedMs;
  if(!$('messages').querySelector('.message'))$('messages').replaceChildren();
  appendMessage('user',text);
  savePending(crypto.randomUUID());
  const response=await fetch('/chat/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text,request_id:pending})});
  const result=await readChatResponse(response,safety=>{
    appendMessage('assistant',safety);
    document.body.dataset.safetyReceivedMs=String(performance.now());
    document.body.dataset.safetyNotices=String(Number(document.body.dataset.safetyNotices||0)+1);
    $('working').textContent='安全提示已显示，正在补充参考信息…';
    $('risk').textContent='高风险 · 优先就医';$('risk').className='badge high';
  });
  document.body.dataset.finalReceivedMs=String(performance.now());
  savePending(null);
  return result;
}
function showMessages(messages){if(!messages.length)return;$('messages').replaceChildren();for(const m of messages){if(m.role==='system')continue;const item=node('article',`message ${m.role}`);item.append(node('div','message-label',m.role==='user'?'你':'Medical Agent'),node('div','message-content',m.content));$('messages').append(item);}$('messages').scrollTop=$('messages').scrollHeight;}
function renderState(state){const risk=state.risk_level||'';$('risk').textContent={HIGH:'高风险 · 优先就医',MEDIUM:'需谨慎关注',LOW:'当前规则低风险',UNKNOWN:'信息待补充'}[risk]||'等待信息';$('risk').className='badge '+(risk==='HIGH'?'high':risk==='LOW'?'low':'');
  const chiefs=state.chief_complaints||[];const candidates=state.chief_complaint_candidates||[];
  $('focus').textContent=`主要咨询：${chiefs.join('、')||'尚未明确'}\n当前焦点：${state.current_focus_label||state.chief_complaint||'尚未明确'}${candidates.length?'\n候选问题：'+candidates.join('、'):''}`;$('focus').style.whiteSpace='pre-wrap';
  $('action').textContent=actions[state.last_action]||state.goal_progress||'等待首轮信息';$('facts').replaceChildren();
  if(state.fact_ledger){
    // Server supplies the authoritative status list through the desktop
    // endpoint below, including old corrections, resolutions and disputes.
    if(Array.isArray(state.desktop_facts))for(const r of state.desktop_facts){const card=node('div',`fact ${r.polarity} ${r.status}`);card.append(node('div','fact-label',labels[r.field]||r.field),node('div','fact-value',(r.polarity==='negated'?'否认：':r.polarity==='uncertain'?'待确认：':'')+r.value+(r.about_label?' · '+r.about_label:'')),node('div','fact-source',`${r.temporality_label} · ${r.subject_label} · ${r.status_label}`),node('div','fact-source',`原文：${r.evidence}`));$('facts').append(card);}
  }else{for(const [field,label] of Object.entries(labels)){const value=state[field];if(value&&(!Array.isArray(value)||value.length)){const card=node('div','fact');card.append(node('div','fact-label',label),node('div','fact-value',Array.isArray(value)?value.join('、'):String(value)));$('facts').append(card);}}}
  if(!$('facts').children.length)$('facts').append(node('p','muted','尚未记录事实'));
  notice(state.fact_extraction_error?(String(state.fact_extraction_error).startsWith('provider:')?
    '模型服务未完成本轮处理，已保留收到的消息和此前事实。请检查上游连接；这不表示你出现了新的症状。':
    '本轮信息未通过可靠性检查，已保留此前事实。请按回复中的问题确认。'):'');
}
async function refresh(){const s=await api('/status');current=s;const extra=await api('/desktop/api/facts');Object.assign(s.case_state,extra);showMessages(s.messages||[]);renderState(s.case_state);await sessions();await trace();return s;}
async function sessions(){const s=await api('/desktop/api/sessions');sessionId=s.current;pending=JSON.parse(localStorage.getItem(pendingKey)||'{}')[sessionId]||null;$('recover').hidden=!pending;$('send').disabled=busy||!!pending;$('sessions').replaceChildren();for(const entry of [...s.items].reverse()){const b=node('button','session'+(entry.id===s.current?' selected':''),entry.title);b.title=entry.title;b.disabled=busy;b.onclick=async()=>{if(busy)return;try{await api('/desktop/api/open',{id:entry.id});await refresh();}catch(e){notice(e.message);}};$('sessions').append(b);if(entry.id===s.current)$('title').textContent=entry.title;}}
async function trace(){const t=await api('/api/debug/trace');$('trace').replaceChildren();for(const e of (t.events||[]).filter(e=>['node_exit','tool_call','risk_assessed','decision_gate'].includes(e.event||e.type)).slice(-40)){const row=node('div','trace-event',nodes[e.node]||e.tool||e.event||e.type);row.append(node('small','',`${e.elapsed_ms!==undefined?e.elapsed_ms+' ms ':''}${e.risk||e.status||''}`));$('trace').append(row);}if(!$('trace').children.length)$('trace').append(node('p','muted','还没有执行记录'));}
function setBusy(value){busy=value;$('send').disabled=value||!!pending;$('new').disabled=value;$('input').disabled=value;$('working').textContent=value?'正在整理信息与选择下一步…':pending?'发送状态待确认 · 请查询，不要重复发送':'Enter 发送 · Shift+Enter 换行';}
$('composer').onsubmit=async e=>{e.preventDefault();const text=$('input').value.trim();if(!text||busy||pending)return;setBusy(true);notice('');try{const reply=await sendChat(text);$('input').value='';await refresh();if(reply.llm_degraded)notice('本轮服务出现降级，结果已标记；请检查服务状态。');}catch(err){if(pending)await recoverPending();else notice(err.message);}finally{setBusy(false);$('input').focus();}};
$('recover').onclick=recoverPending;
$('input').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();$('composer').requestSubmit();}};
$('new').onclick=async()=>{try{await api('/reset',{});$('messages').replaceChildren(node('div','muted','新的问诊已就绪，请描述本次问题。'));$('title').textContent='新问诊';await refresh();}catch(e){notice(e.message);}};
for(const kind of ['facts','trace'])$(kind+'-tab').onclick=()=>{for(const other of ['facts','trace']){$(other+'-tab').classList.toggle('active',kind===other);$(other+'-panel').hidden=kind!==other;}};
$('profile-open').onclick=async()=>{try{const p=await api('/api/profile');$('age').value=p.age||'';$('sex').value=p.sex||'';$('history').value=(p.past_history||[]).join('，');$('profile-notice').textContent='';$('profile-dialog').showModal();}catch(e){notice(e.message);}};
$('profile-close').onclick=()=>$('profile-dialog').close();
$('profile-form').onsubmit=async e=>{e.preventDefault();try{await api('/api/profile',{age:$('age').value,sex:$('sex').value,past_history:$('history').value});$('profile-dialog').close();await refresh();}catch(err){$('profile-notice').textContent=err.message;}};
async function health(){try{const h=await api('/desktop/health');$('dot').className='ready';$('connection').textContent='本机服务已连接';$('mode').textContent=medicalRuntimeLabel(h.mode);if(!busy&&pending&&!recovering)await recoverPending();}catch(e){$('dot').className='';$('connection').textContent='本机服务失联';notice('问诊服务已断开。可从“问诊”菜单重新启动服务。');}}
(async()=>{try{await health();await refresh();if(pending)await recoverPending();document.body.dataset.ready='true';setInterval(health,5000);}catch(e){notice(e.message);}})();
