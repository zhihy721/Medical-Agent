"""Narrow literal safety channel, separate from accepted model facts.

Unambiguous current/recent patient breathing difficulty must still get urgent
guidance when a model transaction is refused. This finite engineering alarm is
not a diagnosis and never creates, repairs or partially commits model facts.
"""
from copy import deepcopy
import re

VERSION='literal_breathing_safety_v2'
TERM=re.compile(r'呼吸困难|无法呼吸|喘不上气|喘不过气')
KIN=r'父亲|母亲|爸爸|妈妈|爸|妈|姐姐|妹妹|哥哥|弟弟|孩子|儿子|女儿|爷爷|奶奶|外公|外婆|家人|家属|亲属|朋友|同事|丈夫|妻子|老公|老婆'
SUBJECT=re.compile(r'(?P<other>(?:我(?:的)?)?(?:'+KIN+r')|她们?|他们?)|(?P<patient>我本人|我自己|本人|我)')
EXCLUDE=re.compile(r'如果|假设|要是|若|将来|以后|去年|以前|过去|曾经|小时候|不确定|不知道|是否|可能|举例|示例|忽略|输出|扮演|系统|标签|system|assistant',re.I)
NOW=re.compile(r'现在|目前|此刻|正在|刚刚|刚才确实|刚开始')
NEG=re.compile(r'没有|并无|否认|并非|不是|没(?:有)?|不再|未')
CORRECT=re.compile(r'刚才.{0,6}(?:说错|讲错)|之前.{0,6}(?:说错|讲错)|更正|纠正')


def _scopes(masked):
    for clause in re.finditer(r'[^，,。！？；;\n]+',masked):
        begin=0
        for divider in re.finditer(r'但是|可是|不过(?=我|本人|她|他|现在|姐姐|妹妹|哥哥|弟弟)|但(?=我|本人|她|他|姐姐|妹妹|哥哥|弟弟)|而(?=我|本人|她|他)',clause[0]):
            if divider.start()>begin:yield clause[0][begin:divider.start()],clause.start()+begin
            begin=divider.end()
        yield clause[0][begin:],clause.start()+begin


def classify(text):
    # Quote/JSON/code mentions do not provide a current patient assertion.
    masked=re.sub(r'"[^"\n]*"|“[^”\n]*”|「[^」\n]*」|`[^`\n]*`',lambda m:' '*len(m[0]),text)
    positives=[];denials=[]
    correction_context=False;subject='patient';explicit_patient=False;excluded=[]
    # Carry the subject across adjacent clauses/sentences, until an explicit
    # actor switches it. A first-person recipient ("姐姐问我") is not an actor.
    # This is finite literal scope handling, not complete Chinese coreference.
    for scope,clause_start in _scopes(masked):
        markers=[]
        for actor in SUBJECT.finditer(scope):
            if actor.lastgroup=='patient' and re.search(r'(?:告诉|问|对|跟|不是在说)$',scope[:actor.start()]):continue
            markers.append(actor)
        clause_subject=markers[-1].lastgroup if markers else subject
        correction_context |= bool(clause_subject=='patient' and CORRECT.search(scope) and (TERM.search(scope) or re.fullmatch(
            r'\s*(?:(?:刚才|之前)(?:我)?(?:说|讲)错了|(?:我)?(?:要)?更正(?:一下)?)\s*',scope)))
        if re.fullmatch(r'\s*(?:还是|又好像|也许|可能)没有\s*',scope):
            if subject=='patient':denials.append(scope)
            continue
        for match in TERM.finditer(scope):
            preceding=[a for a in markers if a.end()<=match.start()]
            actor_subject=preceding[-1].lastgroup if preceding else subject
            actor_explicit=(preceding[-1].lastgroup=='patient') if preceding else explicit_patient
            prefix=scope[:match.start()]
            suffix=scope[match.end():]
            if actor_subject!='patient' or EXCLUDE.search(scope):
                excluded.append(dict(evidence=match[0],start=clause_start+match.start(),end=clause_start+match.end(),subject=actor_subject,reason='other_subject' if actor_subject!='patient' else 'non_current_or_instruction'))
                continue
            if NEG.search(prefix[-6:]) or re.match(r'(?:这(?:个|种))?(?:症状)?(?:不存在|没有)',suffix):
                denials.append(scope);continue
            if NOW.search(scope) or actor_explicit:
                start=clause_start+match.start();end=clause_start+match.end()
                positives.append(dict(concept='呼吸困难',evidence=text[start:end],start=start,end=end))
        if markers:
            subject=markers[-1].lastgroup;explicit_patient=subject=='patient'
    # Same-turn ambiguity does not clear an existing alarm. It is not a new
    # unambiguous positive, either; the usual clarification path remains.
    return dict(positive=positives if not denials else [],
        explicit_correction=bool(denials and correction_context and not positives),
        conflicting=bool(positives and denials),excluded=excluded,
        subject_correction=bool(re.search(r'(?:不是在说我|不是我(?:本人)?|说的是.{0,6}(?:姐姐|妹妹|哥哥|弟弟|父亲|母亲))',masked)))


def _retract_proven_legacy_other(memory,events,signal):
    """Append a correction only when the short original history proves scope.

    Old v1 events lacked source text. Cropped/ambiguous history is insufficient
    to retract an alarm; never undo an unrelated real patient activation.
    """
    if not signal['subject_correction'] or len(memory.history)>=memory.max_history:return []
    start=max((i+1 for i,e in enumerate(events) if e['event']=='explicit_correction'),default=0)
    already={i for e in events if e['event']=='subject_scope_retraction' for i in e['source_event_indices']}
    retract=[]
    for i,event in enumerate(events[start:],start):
        if event['event']!='activate' or i in already:continue
        if event['version']!='literal_breathing_safety_v1':continue
        index=event.get('user_message_index',0)-1
        if not 0<=index<len(memory.history):continue
        role,source=memory.history[index]
        if role!='user' or not event['evidence'] or not all(source[m['start']:m['end']]==m['evidence'] for m in event['evidence']):continue
        scoped=classify(source)
        other_spans={(e['start'],e['end']) for e in scoped['excluded'] if e['reason']=='other_subject'}
        if not scoped['positive'] and all((e['start'],e['end']) in other_spans for e in event['evidence']):retract.append(i)
    if retract:
        events.append(dict(version=VERSION,event='subject_scope_retraction',source_event_indices=retract,
            source='verified_original_user_scope',accepted_fact=False))
    return retract


def apply(memory,text,risk_result):
    signal=classify(text);state=memory.case_state
    events=state.setdefault('input_safety_events',[])
    active=state.get('input_safety_active',False)
    retracted=_retract_proven_legacy_other(memory,events,signal)
    if retracted:
        start=max((i+1 for i,e in enumerate(events) if e['event']=='explicit_correction'),default=0)
        excluded={i for e in events if e['event']=='subject_scope_retraction' for i in e['source_event_indices']}
        active=any(e['event']=='activate' and i not in excluded for i,e in enumerate(events) if i>=start)
    if signal['positive']:
        active=True
        events.append(dict(version=VERSION,event='activate',user_message_index=len(memory.history),
            evidence=signal['positive'],source_text=text,source='literal_user_input',accepted_fact=False))
    elif signal['explicit_correction'] and active:
        active=False
        events.append(dict(version=VERSION,event='explicit_correction',user_message_index=len(memory.history),
            evidence=text,source='literal_user_input',accepted_fact=False))
    state['input_safety_active']=active
    state['input_safety_audit']=dict(version=VERSION,signal=signal,active=active,
        semantic_extraction_success=not bool(state.get('fact_extraction_error')),
        accepted_model_facts_unchanged=True,legacy_other_retractions=retracted)
    result=deepcopy(risk_result)
    if active:
        result['risk']='HIGH'
        result.setdefault('matched_rules',[]).append('literal_patient_breathing_safety')
        result.setdefault('reasons',[]).append('原文含当前/刚发生的呼吸困难安全信号；不依赖模型抽取成功。')
    memory._persist_session_state()
    return result
