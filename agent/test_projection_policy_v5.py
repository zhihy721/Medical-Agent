from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

from agent.factory import create_agent
from agent.semantic_extraction import SemanticExtractor
from agent.test_semantic_runtime import FixedClient,delta
from llm.fact_contract_runtime import V2,build_input
from llm.fact_projection_policy import VERSION,ALIAS_VERSION
from llm.fact_representation import normalize_representation
from llm.fact_clarification import QUESTION
from llm.fact_transaction_guard import check_transactions
from llm.runtime_extraction_guard import classify
from memory.memory import ConversationMemory
from memory.fact_ledger_runtime import statuses,risk_input
from memory.rollout_store import RolloutSessionStore
from training.test_fact_contract import fact


class ProjectionPolicyTests(unittest.TestCase):
    def memory(self):
        m=ConversationMemory();m.enable_fact_mode(V2,representation_policy=VERSION);return m

    def extractor(self,*answers,recovery='off'):
        return SemanticExtractor(FixedClient(*answers),contract_version=V2,
            prompt_version='coverage_v8',representation_mode='finite_v5',recovery_mode=recovery)

    def test_only_exact_values_and_approved_fields(self):
        inp=build_input('喘不上气。便血。黑便。',contract_version=V2)
        d=delta(fact('喘不上气'),fact('便血',id='f2',polarity='uncertain'),fact('黑便',id='f3'))
        before=deepcopy(d)
        new,trace=normalize_representation(d,inp,alias_version=ALIAS_VERSION)
        self.assertEqual(d,before)
        self.assertEqual([(f['field'],f['value']) for f in new['facts']],
            [('red_flags','呼吸困难'),('red_flags','便血'),('symptoms','黑便')])
        self.assertEqual(new['facts'][1]['polarity'],'uncertain')
        self.assertTrue(trace)
        for word in ['疑似喘不上气','活动时喘不上气','黑便','胸痛','胸闷']:
            f=fact(word);d=delta(f)
            output,_=normalize_representation(d,build_input(word,contract_version=V2),alias_version=ALIAS_VERSION)
            self.assertEqual(output['facts'][0]['value'],word)

    def test_v3_v4_and_qualified_or_other_field_values_unchanged(self):
        inp=build_input('喘不上气。',contract_version=V2);d=delta(fact('喘不上气'))
        for version in ['fact_aliases_v3','fact_aliases_v4']:
            new,_=normalize_representation(d,inp,alias_version=version)
            self.assertEqual(new,d)
        d=delta(fact('喘不上气',field='past_history'))
        new,_=normalize_representation(d,inp,alias_version=ALIAS_VERSION)
        self.assertEqual(new,d)

    def test_real_graph_prioritizes_high_and_appends_breathing_question(self):
        ex=self.extractor(delta(fact('喘不过气')))
        with patch('agent.graph._run_guideline',return_value={'advice':[]}):
            agent=create_agent(Mock(),ConversationMemory(),fact_extractor=ex)
            reply=agent.run('喘不过气。')
        self.assertEqual(agent.memory.case_state['risk_level'],'HIGH')
        self.assertIn(QUESTION,reply)
        self.assertIn('不要等待问答',reply)
        self.assertEqual(agent.memory.case_state['fact_ledger']['turns'][0]['extraction_policy_trace']['extractor_raw_delta']['facts'][0]['value'],'喘不过气')

    def test_denial_other_hypothetical_historical_do_not_escalate(self):
        for kw in [dict(polarity='negated'),dict(subject='other'),dict(temporality='hypothetical'),dict(temporality='historical')]:
            m=self.memory();r=self.extractor(delta(fact('喘不上气',**kw))).run(m,'喘不上气。')
            self.assertEqual(r['status'],'ok',r)
            self.assertEqual(risk_input(m.case_state['fact_ledger'])['red_flags'],[])
            self.assertEqual(m.case_state['interpretation_annotations'],[])

    def test_uncertain_blood_does_not_become_affirmed_risk(self):
        m=self.memory();r=self.extractor(delta(fact('便血',polarity='uncertain'))).run(m,'便血。')
        self.assertEqual(r['status'],'ok',r)
        self.assertEqual(risk_input(m.case_state['fact_ledger'])['red_flags'],[])
        self.assertEqual(m.case_state['fact_clarifications'][0]['candidate_concept'],'便血')

    def test_new_breathing_occurrence_and_resolution_preserve_old_denial(self):
        m=self.memory();e=self.extractor(delta(fact('喘不过气','没有喘不过气',polarity='negated')),
            delta(fact('喘不上气','现在开始喘不上气')),
            delta(fact('呼吸困难','呼吸困难现在已经缓解',operation='resolve',temporality='resolved',replaces='t2:f1')))
        for text in ['没有喘不过气。','现在开始喘不上气。','呼吸困难现在已经缓解。']:
            r=e.run(m,text);self.assertEqual(r['status'],'ok',r)
        ledger=m.case_state['fact_ledger'];s=statuses(ledger)
        self.assertEqual((s['t1:f1'],s['t2:f1'],s['t3:f1']),('past','superseded','active'))
        self.assertTrue(risk_input(ledger)['resolved_red_flags'])
        self.assertEqual(ledger['records'][0]['polarity'],'negated')

    def test_explicit_correction_reassesses_breathing_without_resolution(self):
        m=self.memory();e=self.extractor(delta(fact('喘不上气')),
            delta(fact('呼吸困难','我说错了，没有呼吸困难',polarity='negated',operation='correct',replaces='t1:f1')))
        self.assertEqual(e.run(m,'喘不上气。')['status'],'ok')
        r=e.run(m,'我说错了，没有呼吸困难。');self.assertEqual(r['status'],'ok',r)
        self.assertEqual(risk_input(m.case_state['fact_ledger'])['red_flags'],[])

    def test_wrong_other_subject_correction_is_atomic_and_can_repair(self):
        m=self.memory();e=self.extractor(delta(fact('腹痛','父亲有腹痛',subject='other')),
            delta(fact('腹痛','我没有腹痛',polarity='negated',operation='correct',replaces='t1:f1')),
            delta(fact('腹痛','我没有腹痛',polarity='negated')),recovery='guarded_v13')
        self.assertEqual(e.run(m,'父亲有腹痛。')['status'],'ok')
        r=e.run(m,'我没有腹痛。');self.assertEqual(r['status'],'ok',r)
        self.assertEqual(len(r['attempts']),2)
        self.assertEqual(statuses(m.case_state['fact_ledger'])['t1:f1'],'active')
        self.assertEqual(len(m.case_state['fact_ledger']['turns']),2)

    def test_explicit_degree_missing_repaired_without_program_insertion(self):
        m=self.memory();e=self.extractor(delta(fact('牙痛','牙疼得厉害')),
            delta(fact('牙痛','牙疼得厉害'),fact('厉害','牙疼得厉害',field='severity',id='f2',about='f1')),
            recovery='guarded_v13')
        r=e.run(m,'牙疼得厉害。');self.assertEqual(r['status'],'ok',r)
        self.assertEqual(len(r['attempts']),2)
        self.assertIn('no_bound_attribute',r['attempts'][0]['error'])
        self.assertEqual(len(r['attempts'][0]['raw_delta']['facts']),1)
        self.assertEqual(m.case_state['severity'],'厉害')

    def test_calendar_reference_not_duration_and_elapsed_duration_valid(self):
        inp=build_input('上个月腹胀三天。',contract_version=V2)
        for value in ['上个月','去年','昨天']:
            # Use valid evidence for the contract, so failure is semantic type.
            d=delta(fact('腹胀','上个月腹胀三天',temporality='historical'),
                fact(value,'上个月腹胀三天',field='duration',id='f2',about='f1',temporality='historical'))
            with self.assertRaisesRegex(ValueError,'calendar_reference'):check_transactions(inp,d,classify(inp['current_turn']))
        d=delta(fact('腹胀','上个月腹胀三天',temporality='historical'),
            fact('三天',field='duration',id='f2',about='f1',temporality='historical'))
        check_transactions(inp,d,classify(inp['current_turn']))

    def test_current_assert_from_resolution_repaired_to_resolve(self):
        m=self.memory();e=self.extractor(delta(fact('肩痛')),
            delta(fact('肩痛','肩痛现在已经缓解')),
            delta(fact('肩痛','肩痛现在已经缓解',operation='resolve',temporality='resolved',replaces='t1:f1')),
            recovery='guarded_v13')
        self.assertEqual(e.run(m,'肩痛。')['status'],'ok')
        r=e.run(m,'肩痛现在已经缓解。');self.assertEqual(r['status'],'ok',r)
        self.assertEqual(len(r['attempts']),2)
        self.assertEqual(m.case_state['fact_ledger']['records'][-1]['temporality'],'resolved')

    def test_resolution_alarm_abstains_on_negation_quote_hypothesis(self):
        for text in ['肩痛现在没有缓解。','如果肩痛现在已经缓解就再谈。','引用“肩痛现在已经缓解”。']:
            inp=build_input(text,contract_version=V2)
            # This guard is a finite alarm; scope checking still belongs to model.
            kw={'temporality':'hypothetical'} if text.startswith('如果') else {}
            check_transactions(inp,delta(fact('肩痛',text,**kw)),classify(text))

    def test_model_can_normalize_unlisted_word_without_rule_alias(self):
        m=self.memory();r=self.extractor(delta(fact('乏力','全身使不上力'))).run(m,'全身使不上力。')
        self.assertEqual(r['status'],'ok',r)
        self.assertEqual(r['representation_changes'],[])
        self.assertEqual(r['raw_delta']['facts'][0]['evidence'],'全身使不上力')

    def test_version_pair_and_old_session_are_not_migrated(self):
        from llm.fact_lexical_policy import VERSION as OLD
        m=ConversationMemory();m.enable_fact_mode(V2,representation_policy=OLD)
        before=deepcopy(m.case_state)
        with self.assertRaises(ValueError):self.extractor().run(m,'咳嗽。')
        self.assertEqual(m.case_state,before)
        for prompt,mode in [('coverage_v7','finite_v5'),('coverage_v8','finite_v4')]:
            with self.assertRaises(ValueError):SemanticExtractor(None,contract_version=V2,prompt_version=prompt,representation_mode=mode)

    def test_v5_and_raw_source_survive_archive_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=ConversationMemory(session_store=RolloutSessionStore(tmp),session_id='v5',user_id='fixture')
            m.enable_fact_mode(V2,representation_policy=VERSION);m.add_user('喘不上气。')
            self.assertEqual(self.extractor(delta(fact('喘不上气'))).run(m,'喘不上气。')['status'],'ok')
            restored=ConversationMemory(session_store=RolloutSessionStore(tmp),session_id='v5',user_id='fixture')
            restored.enable_fact_mode(V2,representation_policy=VERSION)
            self.assertEqual(restored.case_state['red_flags'],['呼吸困难'])
            self.assertEqual(restored.get_archived_messages()[0]['content'],'喘不上气。')


if __name__=='__main__':unittest.main()
