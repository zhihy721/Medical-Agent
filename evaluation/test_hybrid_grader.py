import unittest
from copy import deepcopy
from evaluation.hybrid_grader import (code_grade, hybrid_result, pair_preference,
    calibration, human_sample, validate_judgment, observe_trajectory_row)


def observation():
    return dict(execution_ok=True, extraction_ok=True, facts=[], risk='LOW', chiefs=[],
        pulse_declined=False, action='ask_followup_single', targets=['severity'], problem_value='头疼')


def rule():
    return dict(id='next',kind='action',severity='major',expected=[
        dict(action='ask_followup_single',target_sets=[['severity'],['duration']],problem_values=['头痛']),
        dict(action='summarize_progress',target_sets=[[]],problem_values=[])])


def judgment(n=2):
    return dict(scores=dict(relevance=n,clarity=n,groundedness=n),issues=[],abstain_reason='')


class HybridTests(unittest.TestCase):
    def test_multiple_actions_and_aliases_pass_but_wrong_target_does_not(self):
        obs=observation()
        self.assertEqual(code_grade(obs,[rule()])['status'],'pass')
        obs.update(action='summarize_progress',targets=[])
        self.assertEqual(code_grade(obs,[rule()])['status'],'pass')
        obs.update(action='ask_followup_single',targets=['severity'],problem_value='腹痛')
        self.assertEqual(code_grade(obs,[rule()])['status'],'fail')

    def test_high_error_cannot_be_offset_by_fluent_quality(self):
        risk=dict(id='high',kind='risk',severity='critical',expected=['HIGH'])
        result=hybrid_result(code_grade(observation(),[risk]),judgment())
        self.assertEqual(result['quality_total_diagnostic_only'],6)
        self.assertEqual(result['status'],'fail')
        self.assertEqual(pair_preference(result,hybrid_result(code_grade(observation(),[rule()]),judgment(1))),'not_comparable')

    def test_untriggered_tool_unknown_and_empty_rules_are_not_success(self):
        r=dict(id='rag',kind='tool_exercised',severity='major',expected='retrieval')
        obs=observation()
        self.assertEqual(code_grade(obs,[r])['status'],'incomplete')
        obs['tools_exercised']=[]
        result=code_grade(obs,[r])
        self.assertEqual(result['checks'][-1]['status'],'not_exercised')
        self.assertEqual(result['status'],'incomplete')
        with self.assertRaises(ValueError):code_grade(obs,[])

    def test_bound_duration_alias_not_wrong_symptom(self):
        expected=dict(field='duration',value='一周',polarity='present',subject='patient',
                      temporality='current',about_value='头痛')
        r=dict(id='duration',kind='fact_present',severity='major',expected=expected)
        obs=observation();obs['facts']=[{**expected,'value':'一个礼拜','about_value':'头疼'}]
        self.assertEqual(code_grade(obs,[r])['status'],'pass')
        obs['facts'][0]['about_value']='腹痛'
        self.assertEqual(code_grade(obs,[r])['status'],'fail')

    def test_judge_suspicion_needs_review_and_abstention_is_not_zero(self):
        j=judgment();j['issues']=[dict(type='factual',severity='major',source='response',
                                     quote='没有头痛',reason='与患者原文冲突')]
        c=code_grade(observation(),[rule()])
        self.assertEqual(hybrid_result(c,j)['status'],'needs_review')
        self.assertEqual(hybrid_result(c,j,confirmed_issues=j['issues'])['status'],'fail')
        abstain=dict(scores=None,issues=[],abstain_reason='上下文不足')
        self.assertEqual(hybrid_result(c,abstain)['status'],'quality_unavailable')
        with self.assertRaises(ValueError):validate_judgment(dict(scores=None,issues=[],abstain_reason=''))

    def test_calibration_keeps_random_targeted_and_empty_denominators_separate(self):
        h={'A':judgment()['scores'],'B':judgment(1)['scores']}
        j=deepcopy(h);j['A']['clarity']=1
        rows=[dict(item_id='r',cohort='random',human=h,judge=h,human_preference='A',judge_preference='A'),
              dict(item_id='t',cohort='targeted',human=h,judge=j,human_preference='A',judge_preference='tie',
                   disagreement_types=['clarity','preference'])]
        self.assertEqual(calibration(rows,'random')['dimension_exact_agreement'],1)
        self.assertEqual(calibration(rows,'targeted')['dimension_mae'],1/6)
        self.assertIsNone(calibration([],'random')['preference_agreement'])

    def test_human_can_dismiss_judge_suspicion_but_not_erase_code_failure(self):
        j=judgment();j['issues']=[dict(type='factual',severity='major',source='response',
                                     quote='没有头痛',reason='疑似冲突')]
        decision=[dict(index=0,decision='dismissed',reviewer='test-human',reason='引用的是患者旧话，并非当前否认')]
        c=code_grade(observation(),[rule()])
        self.assertEqual(hybrid_result(c,j,issue_decisions=decision)['status'],'pass')
        failed=code_grade({**observation(),'problem_value':'腹痛'},[rule()])
        self.assertEqual(hybrid_result(failed,j,issue_decisions=decision)['status'],'fail')

    def test_sampling_does_not_relabel_issue_selection_random_or_drop_overflow(self):
        items=[dict(item_id=str(i),group=str(i%3),review_reasons=['order_conflict']) for i in range(30)]
        first=human_sample(items);again=human_sample(list(reversed(items)))
        self.assertEqual(first,again)
        self.assertEqual(len(first['random']),10);self.assertEqual(len(first['targeted']),5)
        self.assertEqual(len(first['pending_targeted']),15)
        self.assertFalse(set(first['random'])&set(first['targeted']))

    def test_observer_uses_actual_action_and_keeps_missing_tools_unknown(self):
        obs=observe_trajectory_row(dict(status='ok',action_result={'name':'risk_escalation'},
            plan={'next_action':'ask_followup_single','question_targets':[]},case_state={
                'risk_level':'HIGH','chief_complaints':[],'fact_view':{'facts':[]}}))
        self.assertEqual(obs['action'],'risk_escalation')
        self.assertIsNone(obs['tools_exercised'])


if __name__=='__main__':
    unittest.main()
