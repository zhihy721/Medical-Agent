"""Opt-in finite field/lexical view; no original prompt, label or ledger mutation."""
from copy import deepcopy

VERSION='field_policy_alignment_v1'
PROMPT_VERSION='v2_oral_field_prompt_v1'


def field_aligned_prompt(original):
    """One explicit field correction; keep the archived V5 constant unchanged."""
    if original.count('饮水用thirst')!=1:
        raise ValueError('expected exactly one archived oral field clause')
    return original.replace('饮水用thirst','口渴用symptoms，thirst仅作旧结构兼容、不主动输出')


VALUES={
    'symptoms':{'嗓子疼':'咽痛','嗓子痛':'咽痛','嗓子疼痛':'咽痛','鼻子不通气':'鼻塞'},
    'appetite':{'没胃口':'纳差','没胃口，不想吃东西':'纳差'},
    'sweating':{'出汗比平时明显多':'多汗'},
}
PROMPT_OVERRIDE='''
字段政策补充优先于前文：口渴使用symptoms；thirst仅作旧结构兼容，不主动输出。不要把口干当口渴。专用appetite和sweating保留；明确没胃口可用纳差，明确出汗比平时明显多可用多汗，吃得少、单说不想吃东西或仅说出汗不支持这两个值。'''


def align(delta,inp):
    output,context=deepcopy(delta),deepcopy(inp)
    changes=[]
    for scope,rows in [('prediction',output['facts']),('known',context['known_facts'])]:
        for fact in rows:
            before=(fact['field'],fact['value'])
            if before==('thirst','口渴'):fact['field']='symptoms'
            fact['value']=VALUES.get(fact['field'],{}).get(fact['value'],fact['value'])
            if before!=(fact['field'],fact['value']):
                changes.append(dict(scope=scope,id=fact['id'],before=before,
                                    after=(fact['field'],fact['value']),version=VERSION))
    return output,context,changes
