"""Finite v14 rejection alarms; do not change facts or make clinical inferences."""
import re
from llm.fact_transaction_guard import check_transactions
from llm.deferred_scope import evidence_scope,is_deferred_ambiguous

VERSION='compact_transaction_guard_v14'
OBSERVATIONS={'sleep','sweating','appetite','thirst','cold_heat','stool_urine','emotion'}
PROBLEMS={'鼻塞','耳痛','腹痛','肩痛','腿痛','头痛','牙痛','腰痛','咽痛','咳嗽','腹胀'}
ATTRIBUTE_CORRECTION=re.compile(r'的(?:时间|病程|程度|持续时间)(?:刚才|之前|先前)?(?:说错|记错|弄错|填错|写错|报错)')
BLOCK=re.compile(r'如果|假设|要是|引用|转述|[“”「」"]')


def check(inp,delta,classification):
    check_transactions(inp,delta,classification)
    for f in delta['facts']:
        scope=evidence_scope(inp['current_turn'],f['evidence'])
        if BLOCK.search(scope):continue
        if f['field'] in OBSERVATIONS and f['value'] in PROBLEMS:
            raise ValueError('defined_problem_in_unrelated_observation_field')
        if (f['field'] in {'symptoms','red_flags'} and f['polarity']=='negated'
                and f['temporality']=='current' and is_deferred_ambiguous(inp['current_turn'],f['evidence'],modern=True)):
            raise ValueError('deferred_discussion_is_not_symptom_denial')
        if (f['operation']=='correct' and f['field'] not in {'duration','severity','location','pain_character'}
                and f['polarity']=='present' and ATTRIBUTE_CORRECTION.search(scope)):
            # Only reject the redundant object correction if a matching actual
            # old attribute is being corrected too. Ambiguous cases abstain.
            if any(g['operation']=='correct' and g['about'] in {f['id'],f['replaces']}
                    and g['field'] in {'duration','severity'} and
                    any(k['id']==g['replaces'] and k['about']==f['replaces'] for k in inp['known_facts'])
                    for g in delta['facts']):
                raise ValueError('attribute_only_correction_must_not_correct_object')
