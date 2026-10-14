"""Replay observed wrong outputs through real intent/ledger call sites."""
import unittest
from agent.complaint_intent import ComplaintIntent
from agent.semantic_extraction import SemanticExtractor
from agent.test_semantic_runtime import FixedClient, delta
from memory.memory import ConversationMemory
from training.test_fact_contract import fact


def selected_memory():
    m=ConversationMemory();m.enable_fact_mode()
    m.apply_fact_delta(m.fact_input('鼻塞四天，乏力。'),delta(fact('鼻塞'),
        fact('四天',id='f2',field='duration',about='f1'),fact('乏力',id='f3')))
    m.apply_complaint_intent(dict(status='ok',events=[dict(problem_id='t1:f1',operation='select',evidence='主要问鼻塞')]))
    return m


class IntentFactSeamTests(unittest.TestCase):
    def test_choice_context_prompt_routing_does_not_use_patient_text(self):
        from agent.complaint_intent import PROMPTS
        for question, text, expected in [('本次主要想咨询哪个问题？', '乏力', 'v5'),
                ('持续多久？', '本次主要想咨询哪个问题？', 'v4'),
                ('你主要想问鼻塞，请问具体哪个部位？', '乏力', 'v4')]:
            with self.subTest(question=question):
                m = selected_memory(); m.set_fact_question(question)
                client = FixedClient({'events': []})
                audit = ComplaintIntent(client, prompt_version='v4', guard_version='v2',
                    choice_prompt_version='v5').run(m, text)
                self.assertEqual(client.calls[0][0]['content'], PROMPTS[expected])
                self.assertEqual(audit['effective_prompt_version'], expected)

    def test_persistent_symptom_is_not_a_new_consultation(self):
        m=selected_memory();text='乏力的问题还没有消失。'
        client=FixedClient({'events':[dict(problem_id='t1:f3',operation='select',evidence=text)]})
        intent=ComplaintIntent(client,prompt_version='v3',guard_version='v2')
        audit=intent.run(m,text)
        self.assertEqual(m.get_case_state()['chief_complaints'],['鼻塞'])
        self.assertEqual(audit['status'],'error')
        self.assertEqual(len(client.calls),1)

    def test_attribute_uncertainty_is_not_withdrawal(self):
        m=selected_memory();text='还没记清楚具体程度，请先记录不确定。'
        client=FixedClient({'events':[dict(problem_id='t1:f1',operation='withdraw',evidence=text)]})
        intent=ComplaintIntent(client,prompt_version='v3',guard_version='v2')
        audit=intent.run(m,text)
        self.assertEqual(m.get_case_state()['chief_complaints'],['鼻塞'])
        self.assertEqual(audit['status'],'error')

    def test_unapproved_dizziness_rewrite_cannot_commit(self):
        m=ConversationMemory();m.enable_fact_mode()
        wrong=delta(fact('眩晕','头晕'),fact('五天',id='f2',field='duration',about='f1'))
        extractor=SemanticExtractor(FixedClient(wrong,wrong),boundary_version='v4',recovery_mode='guarded_v6')
        result=extractor.run(m,'头晕五天。')
        self.assertEqual(result['status'],'error')
        self.assertFalse(m.get_case_state()['fact_ledger']['records'])

    def test_explicit_select_with_fact_update_is_not_blocked(self):
        m=selected_memory();text='乏力的问题还没有消失，也作为这次重点一起问。'
        client=FixedClient({'events':[dict(problem_id='t1:f3',operation='select',evidence='乏力的问题还没有消失')]})
        audit=ComplaintIntent(client,prompt_version='v4',guard_version='v2').run(m,text)
        self.assertEqual(audit['status'],'ok')
        self.assertEqual(m.get_case_state()['chief_complaints'],['乏力','鼻塞'])

    def test_explicit_withdraw_with_uncertainty_is_not_blocked(self):
        m=selected_memory();text='程度不确定，但先不问鼻塞了。'
        client=FixedClient({'events':[dict(problem_id='t1:f1',operation='withdraw',evidence=text)]})
        audit=ComplaintIntent(client,prompt_version='v4',guard_version='v2').run(m,text)
        self.assertEqual(audit['status'],'ok');self.assertEqual(m.get_case_state()['chief_complaints'],[])

    def test_real_choice_question_allows_persistence_answer(self):
        m=selected_memory();m.set_fact_question('本次主要想咨询哪个问题？')
        text='乏力的问题还没有消失。'
        client=FixedClient({'events':[dict(problem_id='t1:f3',operation='select',evidence=text)]})
        self.assertEqual(ComplaintIntent(client,guard_version='v2').run(m,text)['status'],'ok')

    def test_prior_chief_assertion_does_not_make_an_attribute_question_a_choice(self):
        m=selected_memory();m.set_fact_question('你主要想问鼻塞，请问具体哪个部位？')
        text='乏力的问题还没有消失。'
        client=FixedClient({'events':[dict(problem_id='t1:f3',operation='select',evidence=text)]})
        self.assertEqual(ComplaintIntent(client,guard_version='v2').run(m,text)['status'],'error')
        self.assertEqual(m.get_case_state()['chief_complaints'],['鼻塞'])

    def test_legacy_guard_keeps_original_behavior_and_raw_proposal_is_preserved(self):
        text='乏力的问题还没有消失。';proposal={'events':[dict(problem_id='t1:f3',operation='select',evidence=text)]}
        old=selected_memory();ComplaintIntent(FixedClient(proposal)).run(old,text)
        self.assertIn('乏力',old.get_case_state()['chief_complaints'])
        new=selected_memory();audit=ComplaintIntent(FixedClient(proposal),guard_version='v2').run(new,text)
        self.assertEqual(audit['raw_events'],proposal['events']);self.assertEqual(audit['events'],[])

    def test_literal_dizziness_and_reviewed_headache_alias_are_allowed(self):
        for text,raw in [('眩晕五天。',delta(fact('眩晕'),fact('五天',id='f2',field='duration',about='f1'))),
                         ('头疼两天。',delta(fact('头疼'),fact('两天',id='f2',field='duration',about='f1')))]:
            with self.subTest(text=text):
                m=ConversationMemory();m.enable_fact_mode()
                result=SemanticExtractor(FixedClient(raw),recovery_mode='guarded_v6').run(m,text)
                self.assertEqual(result['status'],'ok')

    def test_repair_retains_head_dizziness_duration_without_program_relabeling(self):
        m=ConversationMemory();m.enable_fact_mode()
        wrong=delta(fact('眩晕','头晕'),fact('五天',id='f2',field='duration',about='f1'))
        right=delta(fact('头晕'),fact('五天',id='f2',field='duration',about='f1'))
        result=SemanticExtractor(FixedClient(wrong),repair_client=FixedClient(right),recovery_mode='guarded_v6').run(m,'头晕五天。')
        self.assertEqual(result['attempts'][0]['raw_delta'],wrong)
        self.assertEqual(result['raw_delta'],right)
        self.assertEqual(m.get_case_state()['fact_view']['problems'][0]['attributes'][0]['value'],'五天')


if __name__=='__main__':unittest.main()
