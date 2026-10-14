"""Evidence-based questions about failed/omitted input, never synthetic facts."""
import hashlib
import re

VERSION='delivery_recovery_v1'


def build(state,text):
    error=str(state.get('fact_extraction_error') or '')
    result=dict(version=VERSION,kind='none',question='',error_code=error,evidence='')
    if error:
        result['kind']='transaction_rejected'
        if error.startswith('provider:'):
            result.update(kind='upstream_failure',question='模型服务未完成本轮处理，已保留收到的文字与此前事实。请检查上游连接后再继续。')
        elif 'occurrence_context_is_not_elapsed_duration' in error:
            result['question']='你描述的是不适出现的情境，还是已经持续的时间？请分别说明什么时候出现、已经持续多久。'
        elif 'explicit_current_denial_conflicts_with_positive' in error:
            result['question']='本轮原文明确否认症状，但模型输出了相反事实，已拒绝写入并保留此前记录。请确认：你现在是否仍没有这个症状？'
        elif 'explicit_current_denial_missing' in error:
            result['question']='你刚才是在明确否认提到的症状吗？本轮否认未能可靠写入记录，请再确认一次；这不表示你出现了这些症状。'
        elif 'literal_positive_condition_missing_scoped_fact' in error:
            result['question']='你提到的不适是现在或之前确实发生，还是仅假设以后出现？请确认主体和时间。'
        elif 'red_flag_without_concept_evidence' in error:
            result['question']='本轮出现了没有原文依据的风险推断，已拒绝写入。请用自己的话分别描述实际不适及什么时候出现。'
        else:result['question']='本轮信息未通过可靠性检查，尚未更新事实。请分别说明是谁、有什么不适，以及时间或更正关系。'
        return result
    for clause in re.split(r'[，。；,;!?！？]',text or ''):
        match=re.fullmatch(r'(?:我)?(?P<term>心悸|心慌|耳痛|头痛|头疼|咳嗽|鼻塞|腰痛|腹痛|呼吸困难)(?:的事|的问题|的情况)?(?:以后再谈|以后再说|回头再谈|回头再说|先不谈|先不讨论)',clause.strip())
        if not match:continue
        term=match['term']
        from llm.fact_normalization import canonical_value
        concept=canonical_value('symptoms',term)
        if any(p['value']==concept for p in (state.get('fact_view') or {}).get('problems',[])):continue
        if state.get('fact_clarifications'):continue
        key=hashlib.sha256(concept.encode()).hexdigest()[:16]
        if state.get('delivery_clarification_counts',{}).get(key,0)>=2:continue
        return dict(version=VERSION,kind='missing_deferred_candidate',error_code='',evidence=clause.strip(),
                    key=key,question=f'你现在有{term}吗，还是只是想以后讨论？已记录的其他事实不会因延期讨论被删除。')
    return result
