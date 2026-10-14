"""Narrow literal safety channel, separate from accepted model facts.

Unambiguous current/recent patient breathing difficulty must still get urgent
guidance when a model transaction is refused. This finite engineering alarm is
not a diagnosis and never creates, repairs or partially commits model facts.
"""
from copy import deepcopy
import re

VERSION='literal_breathing_safety_v1'
TERM=re.compile(r'呼吸困难|无法呼吸|喘不上气|喘不过气')
OTHER=re.compile(r'父亲|母亲|爸爸|妈妈|我爸|我妈|孩子|儿子|女儿|朋友|同事|丈夫|妻子')
EXCLUDE=re.compile(r'如果|假设|要是|若|将来|以后|去年|以前|过去|曾经|小时候|不确定|不知道|是否|可能|举例|示例|忽略|输出|扮演|系统|标签|system|assistant',re.I)
NOW=re.compile(r'现在|目前|此刻|正在|刚刚|刚才确实|刚开始')
NEG=re.compile(r'没有|并无|否认|并非|不是|没(?:有)?|不再|未')
CORRECT=re.compile(r'刚才.{0,6}(?:说错|讲错)|之前.{0,6}(?:说错|讲错)|更正|纠正')


def classify(text):
    # Quote/JSON/code mentions do not provide a current patient assertion.
    masked=re.sub(r'"[^"\n]*"|“[^”\n]*”|「[^」\n]*」|`[^`\n]*`',lambda m:' '*len(m[0]),text)
    positives=[];denials=[]
    correction_context=False
    for clause in re.finditer(r'[^，,。！？；;\n]+',masked):
        scope=clause[0]
        correction_context |= bool(CORRECT.search(scope) and (TERM.search(scope) or re.fullmatch(
            r'\s*(?:(?:刚才|之前)(?:我)?(?:说|讲)错了|(?:我)?(?:要)?更正(?:一下)?)\s*',scope)))
        if re.fullmatch(r'\s*(?:还是|又好像|也许|可能)没有\s*',scope):
            denials.append(scope)
            continue
        if OTHER.search(scope) or EXCLUDE.search(scope):continue
        for match in TERM.finditer(scope):
            prefix=scope[:match.start()]
            suffix=scope[match.end():]
            if NEG.search(prefix[-6:]) or re.match(r'(?:这(?:个|种))?(?:症状)?(?:不存在|没有)',suffix):
                denials.append(scope);continue
            if NOW.search(scope):
                start=clause.start()+match.start();end=clause.start()+match.end()
                positives.append(dict(concept='呼吸困难',evidence=text[start:end],start=start,end=end))
    # Same-turn ambiguity does not clear an existing alarm. It is not a new
    # unambiguous positive, either; the usual clarification path remains.
    return dict(positive=positives if not denials else [],
        explicit_correction=bool(denials and correction_context and not positives),
        conflicting=bool(positives and denials))


def apply(memory,text,risk_result):
    signal=classify(text);state=memory.case_state
    events=state.setdefault('input_safety_events',[])
    active=state.get('input_safety_active',False)
    if signal['positive']:
        active=True
        events.append(dict(version=VERSION,event='activate',user_message_index=len(memory.history),
            evidence=signal['positive'],source='literal_user_input',accepted_fact=False))
    elif signal['explicit_correction'] and active:
        active=False
        events.append(dict(version=VERSION,event='explicit_correction',user_message_index=len(memory.history),
            evidence=text,source='literal_user_input',accepted_fact=False))
    state['input_safety_active']=active
    state['input_safety_audit']=dict(version=VERSION,signal=signal,active=active,
        semantic_extraction_success=not bool(state.get('fact_extraction_error')),
        accepted_model_facts_unchanged=True)
    result=deepcopy(risk_result)
    if active:
        result['risk']='HIGH'
        result.setdefault('matched_rules',[]).append('literal_patient_breathing_safety')
        result.setdefault('reasons',[]).append('原文含当前/刚发生的呼吸困难安全信号；不依赖模型抽取成功。')
    memory._persist_session_state()
    return result
