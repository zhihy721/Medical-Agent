"""Versioned interpretation annotations used by online risk and response layers.

Never rewrites a fact or treats lexical ambiguity as patient uncertainty. The
runtime integration must preserve existing HIGH response priority.
"""
VERSION = 'breathing_interpretation_v1_draft'
QUESTION = ('你说喘不上气，具体是什么感觉？是胸口发闷、吸气费力，还是觉得空气不够？'
            '这些感觉也可能同时存在。现在能正常说完整句话吗？')


def breathing_clarifications(records):
    annotations = []
    for fact in records:
        if (fact['subject'] != 'patient' or fact['temporality'] != 'current'
                or fact['polarity'] == 'negated'
                or fact['field'] not in {'symptoms', 'red_flags'}
                or fact['value'] not in {'呼吸困难', '胸闷', '气短'}):
            continue
        evidence = fact['evidence']
        if not ('气' in evidence and any(p in evidence for p in ('喘不上', '喘不过'))):
            continue
        annotations.append({'fact_id': fact['id'], 'candidate_concept': '呼吸困难',
                            'interpretation_status': 'needs_clarification',
                            'safety_candidate': True, 'question': QUESTION,
                            'policy_version': VERSION,
                            'reason': 'colloquial_breathing_description',
                            'source_polarity': fact['polarity']})
    return annotations
