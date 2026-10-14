"""Finite delivery polarity alarm; reject, never relabel or change risk.

Historical/current mixed clauses, questions, quotes and double negation need
semantic interpretation. Abstain there; passing is not semantic approval.
"""
import re
from llm.deferred_scope import evidence_scope
from llm.fact_normalization import canonical_value
from llm.fact_projection_policy import ALIAS_VERSION, BREATHING_WORDS
from llm.runtime_extraction_guard import CLINICAL

VERSION='delivery_polarity_guard_v1'
DENIAL=re.compile(r'(?:没有|否认)(?:任何|明显的?)?\s*$')
AMBIGUOUS=re.compile(r'如果|假设|要是|引用|转述|[“”「」\"]')
QUESTION_OR_DOUBLE=re.compile(r'不确定|不知道|是否|是不是|说不准|并非|不是|不能说')


def check(inp,delta):
    for item in delta['facts']:
        if not (item['field'] in {'symptoms','red_flags'} and item['subject']=='patient'
                and item['polarity']=='present' and item['temporality']=='current'):
            continue
        scope=evidence_scope(inp['current_turn'],item['evidence'])
        if AMBIGUOUS.search(scope):continue
        concept=canonical_value(item['field'],item['value'],version=ALIAS_VERSION)
        mentions=[m for m in CLINICAL.finditer(scope)
                  if BREATHING_WORDS.get(m.group(),canonical_value('symptoms',m.group(),version=ALIAS_VERSION))==concept]
        # Do not treat a past denial and a new occurrence as the same assertion.
        # Evidence containing two mentions is deliberately outside this alarm.
        if len(mentions)!=1:continue
        prefix=scope[:mentions[0].start()]
        denial=DENIAL.search(prefix)
        if not denial:continue
        before=prefix[:denial.start()]
        if before.endswith('有') or QUESTION_OR_DOUBLE.search(before):continue
        raise ValueError('explicit_current_denial_conflicts_with_positive_fact:'+item['id'])
