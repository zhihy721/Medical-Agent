"""Literal evaluation aliases only. No runtime, risk, or old scores are changed."""
import re
from llm.fact_normalization import canonical_value

VERSION='agent_eval_lexical_aliases_v1'
ALIASES={
    'symptoms': {'肩膀痛':'肩痛','肩膀疼':'肩痛','肩疼':'肩痛',
        '喉咙疼':'咽痛','喉咙痛':'咽痛','嗓子疼':'咽痛','嗓子痛':'咽痛','咽疼':'咽痛',
        '鼻子不通':'鼻塞'},
    'severity': {'轻微':'轻度','偏轻':'轻度'},
    'sleep': {'睡觉多梦':'多梦'},
}
NUMBERS={'一':1,'二':2,'两':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10}


def normalize(field,value):
    value=canonical_value(field,value)
    value=ALIASES.get(field,{}).get(value,value)
    if field=='duration' and isinstance(value,str):
        match=re.fullmatch(r'([一二三四五六七八九十两]|[1-9][0-9]?)天',value)
        if match:
            number=match.group(1)
            return str(NUMBERS.get(number,number))+'天'
    return value
