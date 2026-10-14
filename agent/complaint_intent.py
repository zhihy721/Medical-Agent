"""Independent, evidence-backed consultation intent; never a risk input."""
from copy import deepcopy
from dataclasses import asdict
import json
import re

VERSION = 'complaint_intent_v1_draft'
PROMPT = '''识别患者本次重点咨询的意图，输出JSON events列表；没有明确意图时为空。
每项严格包含problem_id、operation(select或withdraw)、evidence(当前轮唯一连续原文)。
只能引用输入problems的ID。可选择多个，不根据症状出现顺序推断主次。
“主要想问头痛和咳嗽”选择两者；“头痛伴咳嗽”不选择。
只有last_question明确询问重点咨询什么时，简短回答症状才可作为选择依据。
撤回重点咨询只withdraw，不等于否认症状。不要诊断、修改事实或评价风险。
此前选择不必重复输出。患者未谈咨询重点时不要撤回旧选择。
这是咨询优先意图，不是症状抽取。仅有症状或时长，即使是多个，也输出{"events":[]}。
例：当前轮“咳嗽三天，头痛一天。”、last_question为空 → {"events":[]}。
例：当前轮“头痛”、last_question“持续多久？” → {"events":[]}。
只有明确“主要想问/重点咨询/先谈这个”，或回答明确的咨询重点问题，才select。
problems为空时必须输出{"events":[]}，不能生成null或新的ID。'''

PROMPTS = {'v1': PROMPT, 'v2': PROMPT + '''
同一轮可以同时select新重点和withdraw旧重点；不能只输出新select而遗漏明确撤回。
"不重点问/不想重点咨询"某问题只撤回咨询重点，绝不把该症状标为否认。
同义症状按problems的规范值找到ID；例如头疼对应头痛。只能使用实际输入ID。
例：problems=[{"id":"p1","value":"头痛"},{"id":"p2","value":"咳嗽"}]，此前selected含p1。
当前轮“改成重点问咳嗽，不重点问头疼了。” → {"events":[{"problem_id":"p2","operation":"select","evidence":"重点问咳嗽"},{"problem_id":"p1","operation":"withdraw","evidence":"不重点问头疼了"}]}。
同一输入如果仅说“咳嗽又多了一些”，没有撤回或选择重点 → {"events":[]}。
新增重点不隐含撤回其他重点；只有当前轮明确撤回时才withdraw。
上述p1/p2仅为示例，必须替换为当前输入实际problem_id。'''}

# Keep v1/v2 reproducible. v3 is a development candidate, not a route promotion.
PROMPTS['v3'] = PROMPTS['v2'] + '''
先判断当前轮是否真的在改变“咨询重点”，再输出事件；不能因补充、否认、纠错或确认症状事实就选择或撤回咨询重点。
“没有胸痛，也没有呼吸困难”“口渴的问题还没有消失”“恶心还是两天，没有说它消失”都是事实陈述，不改变旧重点，events为空。
“不用撤掉/不要撤掉/不必撤掉”是保持旧重点，绝不是withdraw；可以只输出同句中新重点的select。
例：此前selected有肩痛，当前轮“乏力也要作为这次主要问题一起问，不用撤掉肩痛。”只select乏力，保留肩痛，不输出肩痛withdraw。
例：此前selected有咽痛，当前轮“恶心还是两天，没有说它消失。” events为空，咽痛仍是原重点。
只有“不再重点咨询/先不问/撤掉该重点”等明确咨询撤回，才withdraw。“没有/没消失/时间没变化”本身不是咨询撤回。
不能将事实问句（程度、病程、部位、是否有某症状）误当重点选择问句。
当前轮对旧重点未提出操作，就保持其selected；不是把所有problems重新排序或重选一遍。
'''

PROMPTS['v4'] = PROMPTS['v3'] + '''
事实持续存在和事实属性不确定都不是咨询选择。不要因为句子中出现“问题”就select。
“乏力的问题还没有消失”仅补充症状状态，不表示主要想问乏力，events为空。
“还没记清楚具体程度，请先记录不确定”仅表达程度不确定，不撤回任何旧咨询重点，events为空。
“鼻塞时间没有变化”“先记录病程不确定”同样不改变重点。
与之区别：“乏力还没消失，也作为这次重点一起问”可select乏力；“程度没记清，但先不问鼻塞了”可withdraw鼻塞。
一个事实更新只有同时明确咨询操作，或确实回答上一条咨询选择问句，才支持事件。不要通过删除事实或改变风险来解决意图问题。
'''

PROMPTS['v5'] = PROMPTS['v4'] + '''
last_question只能帮助解释当前回答，绝不能作为evidence。evidence必须是current_turn里的唯一连续原文。
例：problems=[{"id":"x","value":"乏力"}]，last_question="本次主要想咨询哪个问题？"，current_turn="乏力"。
正确输出为{"events":[{"problem_id":"x","operation":"select","evidence":"乏力"}]}，不是以问句作evidence。
换用输入中的实际ID；短回答不要求扩写成“主要想问乏力”。只选择回答实际指向的症状，不隐含撤回未明确撤回的旧重点。
'''


def is_choice_question(question):
    """Finite context recognizer, separate from patient intent classification."""
    return any(re.search(r'(?:主要|重点|优先)(?:想|要)?(?:问|咨询|聊|谈|处理).{0,8}(?:哪个|哪些|什么|哪一)', clause)
               for clause in re.split(r'[，。？?]', question or ''))


def select_prompt_version(inp, default_version, choice_version=None):
    return choice_version if choice_version and is_choice_question(inp.get('last_question', '')) else default_version


def check_fact_only_events(inp, events):
    """Finite contradictory-evidence alarm; never creates/selects an event.

    It catches observed state/attribute statements only. This is not a complete
    intent classifier or prompt-injection defence. Reject atomically and retain
    previous selections; raw model output is kept for auditing.
    """
    text=inp['current_turn']
    consultation=re.compile(r'主要|重点|优先|先聊|先谈|先问|先咨询|这次想问|撤掉|撤回|不问|不谈')
    choice_question=is_choice_question(inp.get('last_question',''))
    if consultation.search(text) or choice_question:
        return
    for event in events:
        evidence=event['evidence']
        if re.search(r'(?:问题)?还没(?:有)?消失|(?:时间|病程).{0,5}(?:没(?:有)?变化|没变)',evidence):
            raise ValueError('fact_persistence_is_not_consultation_evidence')
        if re.search(r'(?:没(?:有)?记清楚|没想起来|不确定).{0,8}(?:程度|病程|时间)|(?:程度|病程|时间).{0,8}不确定',evidence):
            raise ValueError('attribute_uncertainty_is_not_consultation_evidence')


def intent_input(state, text):
    return {'current_turn': text, 'last_question': state.get('semantic_last_question', ''),
            'problems': [{'id': p['id'], 'value': p['value']} for p in state['fact_view']['problems']],
            'selected': deepcopy(state.get('complaint_intent_selected', []))}


def validate_events(inp, output):
    if not isinstance(output, dict) or set(output) != {'events'} or not isinstance(output['events'], list):
        raise ValueError('intent requires events list')
    ids = {p['id'] for p in inp['problems']}
    if len(output['events']) > len(ids):
        raise ValueError('too many intent events')
    seen = set()
    for event in output['events']:
        if not isinstance(event, dict) or set(event) != {'problem_id', 'operation', 'evidence'}:
            raise ValueError('invalid intent event fields')
        if not all(isinstance(v, str) for v in event.values()):
            raise ValueError('intent values must be strings')
        if event['problem_id'] not in ids or event['problem_id'] in seen:
            raise ValueError('unknown or duplicate intent problem')
        if event['operation'] not in {'select', 'withdraw'}:
            raise ValueError('invalid intent operation')
        if not event['evidence'] or inp['current_turn'].count(event['evidence']) != 1:
            raise ValueError('intent evidence must occur exactly once')
        seen.add(event['problem_id'])
    return deepcopy(output['events'])


class ComplaintIntent:
    def __init__(self, client, *, prompt_version='v1', guard_version='v1', choice_prompt_version=None, structured_mode='off'):
        if prompt_version not in PROMPTS:
            raise ValueError('invalid intent prompt_version')
        if guard_version not in {'v1','v2'}:
            raise ValueError('invalid intent guard_version')
        if choice_prompt_version is not None and choice_prompt_version not in PROMPTS:
            raise ValueError('invalid choice prompt_version')
        if structured_mode not in {'off','whole_turn_v1'}:
            raise ValueError('invalid intent structured_mode')
        self.client = client
        self.prompt_version = prompt_version
        self.guard_version = guard_version
        self.choice_prompt_version = choice_prompt_version
        self.structured_mode = structured_mode

    def run(self, memory, text):
        inp = intent_input(memory.get_case_state(), text)
        effective_prompt = select_prompt_version(inp, self.prompt_version, self.choice_prompt_version)
        audit = {'version': VERSION if self.prompt_version=='v1' else 'complaint_intent_'+self.prompt_version,
                 'input': inp, 'status': 'error', 'events': [], 'guard_version': self.guard_version,
                 'effective_prompt_version': effective_prompt, 'choice_prompt_version': self.choice_prompt_version,
                 'structured_mode': self.structured_mode}
        try:
            kwargs={'require_json':True}
            if self.structured_mode == 'whole_turn_v1':
                kwargs['json_schema']=intent_schema(inp)
                audit['output_schema']=deepcopy(kwargs['json_schema'])
            result = self.client.call([{'role': 'system', 'content': PROMPTS[effective_prompt]},
                {'role': 'user', 'content': json.dumps(inp, ensure_ascii=False)}], **kwargs)
            audit['model_result'] = asdict(result)
            if not result.ok:
                raise ValueError('provider:' + result.status)
            parsed=json.loads(result.content)
            events=validate_events(inp, parsed)
            audit['raw_events']=deepcopy(events)
            if self.guard_version == 'v2':
                check_fact_only_events(inp,events)
            audit['events'] = events
            audit['status'] = 'ok'
        except (ValueError, TypeError, KeyError) as exc:
            audit['error'] = str(exc)
        memory.apply_complaint_intent(audit)
        return audit


def intent_schema(inp):
    """Constrain shape/IDs/literal evidence, never prescribe intent decisions.

    Evidence is the entire current utterance. It remains patient data, not a
    reasoning chain or program-selected semantic span. Model selects events.
    """
    ids=list(dict.fromkeys(p['id'] for p in inp['problems']))
    events={'type':'array','maxItems':len(ids)}
    if not ids:
        events['items']={'type':'object'}
    else:
        events['items']={'type':'object','additionalProperties':False,
            'required':['problem_id','operation','evidence'],
            'properties':{'problem_id':{'type':'string','enum':ids},
                'operation':{'enum':['select','withdraw']},
                'evidence':{'type':'string','const':inp['current_turn']}}}
    return {'type':'object','additionalProperties':False,'required':['events'],'properties':{'events':events}}
