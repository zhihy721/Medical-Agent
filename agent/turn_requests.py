"""Finite, explicit interaction requests; never medical fact extraction."""
import re
from copy import deepcopy

VERSION='delivery_turn_requests_v1'


def parse(text):
    result={'version':VERSION,'summary':False,'references':False,'pulse_declined':False,'evidence':[]}
    for clause in re.split(r'[，。；,;!?！？]',text or ''):
        clause=clause.strip()
        # Quoted/instruction-like/hypothetical/other-person requests have no
        # authority to alter routing. This is a bounded recognizer, not NLU.
        if re.search(r'[“”"「」]|如果|假如|假设|系统|开发者|忽略|姐姐|哥哥|爸爸|妈妈|他|她|朋友',clause):continue
        if re.fullmatch(r'(?:我)?(?:不做脉诊|不想做脉诊|不测脉搏|先不测脉搏)',clause):
            result['pulse_declined']=True;result['evidence'].append(clause)
        if re.search(r'^(?:请|请帮我|帮我|我想|我希望)(?:根据|把|先|现在|直接)?(?:目前|当前|现在|已知|已有|现有|之前|这些|对话|信息|症状|记录|本次|的|情况|内容|问诊|整理|总结|一下|进展|给我|做个|做一个|汇总|归纳)+',clause) and re.search(r'整理|总结|汇总|归纳',clause):
            if not re.search(r'不要|不用|不必|不想',clause):result['summary']=True;result['evidence'].append(clause)
        if re.search(r'使用资料|查(?:询|找|阅)?(?:资料|知识库)|引用|标明来源|注明来源|附上来源',clause):
            if not re.search(r'不要|不用|不必|不想',clause):result['references']=True;result['evidence'].append(clause)
    return result


def update(memory,text):
    value=parse(text)
    memory.case_state['delivery_turn_request']=value
    if value['pulse_declined']:memory.mark_pulse_skipped()
    return value


def summary_plan(plan,request):
    result=deepcopy(plan)
    result.update(next_action='summarize_progress',next_focus='pattern_summary',
                  question_targets=[],question_target_fields=[],pending_questions=[],
                  action_reason='按患者明确要求整理当前事实与尚未确认的信息。',
                  requested_progress=True,requested_references=bool(request.get('references')))
    return result
