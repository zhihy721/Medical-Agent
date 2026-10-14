"""Delivery v2 literal alarms. Reject transactions; never invent/repair labels.

This is a conservative, finite check, not a semantic oracle. A rejection keeps
the previous ledger intact and requests clarification. V16 evidence is frozen.
"""
import re
from llm.deferred_scope import evidence_scope

VERSION = 'delivery_grounding_guard_v1.1'
OTHER = re.compile(r'父亲|母亲|爸爸|妈妈|我爸|我妈|孩子|儿子|女儿|朋友|哥们儿|同事|丈夫|妻子')
SELF = re.compile(r'我自己|我(?:现在|目前|今|这回|确实|没有|有|是|感到|觉得)')
RISK_TERMS = {
    '呼吸困难': (r'呼吸困难|无法呼吸|喘不上气|喘不过气|气短|呼吸(?:特别|很|非常)?费劲',),
    '胸痛伴呼吸困难': (r'胸(?:痛|疼)', r'呼吸困难|无法呼吸|喘不上气|喘不过气|气短|呼吸(?:特别|很|非常)?费劲'),
    '持续胸痛': (r'胸(?:痛|疼)', r'持续|一直|不停|不间断'),
    '突发剧烈头痛': (r'头(?:痛|疼)', r'突(?:发|然)|一下子', r'剧烈|非常|厉害|很重'),
    '意识障碍': (r'意识|昏迷|昏厥|神志|叫不醒|不省人事',),
    '便血': (r'便血|大便.{0,6}(?:血|出血)|血便|便中带血',),
}


def check(inp, delta):
    from llm.delivery_polarity_guard import check as check_polarity
    check_polarity(inp, delta)
    known = {f['id']: f for f in inp['known_facts']}
    for f in delta['facts']:
        scope = evidence_scope(inp['current_turn'], f['evidence'])
        # Limit this alarm to explicit attribution. Unidentified speakers or
        # mixed clauses cannot be resolved by this rule and remain model errors.
        if OTHER.search(scope) and not SELF.search(scope) and f['subject'] == 'patient':
            raise ValueError('delivery_explicit_other_as_patient:' + f['id'])
        if f['field'] == 'red_flags' and f['value'] in RISK_TERMS:
            old = known.get(f.get('replaces'))
            linked = (f['operation'] in {'correct', 'resolve'} and old is not None
                      and old['value'] == f['value'] and old['subject'] == f['subject'])
            if not linked and not all(re.search(pattern, scope) for pattern in RISK_TERMS[f['value']]):
                raise ValueError('delivery_red_flag_without_concept_evidence:' + f['id'])
        # Reject a clear body-region contradiction; do not normalize unseen
        # wording or infer a diagnosis from a body location alone.
        if (f['field'] == 'symptoms' and f['operation'] == 'assert'
                and f['value'] == '腹痛' and re.search(r'后背(?:疼|痛)', scope)
                and not re.search(r'肚子|腹', scope)):
            raise ValueError('delivery_body_region_contradiction:' + f['id'])
