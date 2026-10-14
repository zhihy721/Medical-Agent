"""Prospective exact-value aliases and clarification policy; no clinical inference."""
import re
VERSION = 'semantic_representation_v4'
ALIAS_VERSION = 'fact_aliases_v4'
# Complete values only. A qualified or unidentified body region stays untouched.
BODY = {'头': '头', '肩膀': '肩', '肩部': '肩', '膝盖': '膝', '颈部': '颈',
        '脖子': '颈', '手': '手', '腿': '腿', '腰': '腰', '后背': '背',
        '背部': '背', '耳朵': '耳', '牙': '牙', '肚子': '腹'}
PAIN_ALIASES = {body + suffix: canonical + '痛'
                for body, canonical in BODY.items() for suffix in ('痛', '疼', '疼痛')}
# User-approved complete symptom values only. Never strip qualifiers or use
# these mappings for emotion, risk categories, or arbitrary lexical fragments.
APPROVED_SYMPTOM_ALIASES = {'没劲儿': '乏力', '喉咙疼': '咽痛'}
HEART_QUESTION = '你说的“心慌”具体是什么感觉：心跳明显、加快或不规则，还是紧张不安，或者其他感觉？这些感觉也可能同时存在。'


def check_heart_rewrite(delta):
    """Reject one unsupported conversion, never repair or infer a new fact."""
    for f in delta['facts']:
        if f['value'] == '心悸' and '心慌' in f['evidence'] and '心悸' not in f['evidence']:
            raise ValueError('heart_expression_requires_original_word_and_clarification')


def check_degree_rewrite(delta):
    for f in delta['facts']:
        if (f['field'] == 'severity' and f['value'] in {'重度', '很重', '剧烈'}
                and '厉害' in f['evidence'] and not re.search(r'重度|很重|剧烈', f['evidence'])):
            raise ValueError('degree_expression_requires_original_厉害_not_重度')
        if (f['field'] == 'severity' and f['value'] in {'轻度', '很轻', '轻微', '轻'}
                and '轻多了' in f['evidence']
                and not re.search(r'轻度|很轻|轻微', f['evidence'])):
            raise ValueError('relative_degree_requires_original_轻多了_not_轻度')


def check_pain_negation(delta):
    """Finite literal alarm for the added aliases; no polarity repair or inference."""
    for f in delta['facts']:
        if not (f['field'] == 'symptoms' and f['polarity'] == 'present'
                and f['subject'] == 'patient' and f['temporality'] == 'current'):
            continue
        names = {word for word, value in PAIN_ALIASES.items() if value == f['value']}
        if any(re.search(r'(?:明确没有|我没有|否认|并无|没有)' + re.escape(name), f['evidence'])
               for name in names):
            raise ValueError('new_pain_alias_literal_negation_conflicts_with_positive_fact')


def heart_interpretations(view, ledger):
    annotations = []
    for f in view['facts']:
        if not (f.get('derived_from') is None and f['field'] in {'symptoms', 'emotion'}
                and f['value'] == '心慌' and f['subject'] == 'patient'
                and f['temporality'] == 'current' and f['polarity'] in {'present', 'uncertain'}):
            continue
        item = dict(fact_id=f['id'], candidate_concept='心慌', kind='heart_interpretation',
                    question=HEART_QUESTION, policy_version=VERSION, status='needs_clarification')
        for turn in ledger['turns']:
            inp = turn.get('input') or {}
            if (inp.get('last_question') != HEART_QUESTION or inp.get('turn_id', 0) <= f['source_turn']
                    or not re.search(r'指的是|就是|只是|说的心慌.{0,4}是', inp.get('current_turn', ''))):
                continue
            candidates = (turn.get('normalized_delta') or turn.get('raw_delta') or {}).get('facts', [])
            explicit = [g for g in candidates if (g['field'], g['value']) in {
                        ('emotion', '紧张'), ('emotion', '焦虑'), ('symptoms', '心悸')}
                        and g['subject'] == 'patient' and g['temporality'] == 'current'
                        and g['polarity'] == 'present' and g['value'] in g['evidence']]
            if explicit:
                item.update(status='clarified', answer_turn=inp['turn_id'],
                            answer_facts=explicit, question='')
        annotations.append(item)
    return annotations
