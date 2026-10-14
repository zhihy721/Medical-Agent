"""Real extractor/ledger/graph regression; synthetic clients, no model scores."""
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

from agent.factory import create_agent
from agent.semantic_extraction import SemanticExtractor
from agent.test_semantic_runtime import FixedClient, delta
from llm.fact_contract_runtime import V2, build_input, position_delta
from llm.fact_representation import normalize_representation
from memory.memory import ConversationMemory
from memory.fact_ledger import project, statuses
from training.test_fact_contract import fact


class RepresentationTests(unittest.TestCase):
    def run_fact(self, text, *facts, memory=None):
        memory = memory or ConversationMemory()
        memory.enable_fact_mode(V2)
        client = FixedClient(delta(*facts), delta(*facts))
        audit = SemanticExtractor(client, contract_version=V2, recovery_mode='guarded_v10',
                                  prompt_version='coverage_v3').run(memory, text)
        return memory, audit

    def test_sleep_fields_and_duration_keep_raw_output(self):
        m, a = self.run_fact('耳痛四天，多梦八天。', fact('耳痛'),
            fact('四天', field='duration', id='f2', about='f1'), fact('多梦', id='f3'),
            fact('八天', field='duration', id='f4', about='f3'))
        self.assertEqual(a['status'], 'ok')
        self.assertEqual(a['raw_delta']['facts'][2]['field'], 'symptoms')
        self.assertEqual(m.case_state['sleep'], '多梦')
        self.assertEqual([p['value'] for p in project(m.case_state['fact_ledger'])['problems']], ['耳痛'])
        self.assertEqual(m.case_state['observation_attributes'][0]['attributes'][0]['value'], '八天')

    def test_sleep_duration_can_be_corrected_without_changing_ear(self):
        m, _ = self.run_fact('耳痛四天，多梦八天。', fact('耳痛'),
            fact('四天', field='duration', id='f2', about='f1'), fact('多梦', field='sleep', id='f3'),
            fact('八天', field='duration', id='f4', about='f3'))
        m, a = self.run_fact('多梦其实十天，刚才记错了。',
            fact('十天', '多梦其实十天，刚才记错了', field='duration', about='t1:f3',
                 operation='correct', replaces='t1:f4'), memory=m)
        self.assertEqual(a['status'], 'ok')
        self.assertEqual(statuses(m.case_state['fact_ledger'])['t1:f4'], 'superseded')
        self.assertEqual(m.case_state['duration'], '四天')
        self.assertEqual(m.case_state['observation_attributes'][0]['attributes'][0]['value'], '十天')

    def test_negative_value_and_cold_heat_field_are_canonical(self):
        m, a = self.run_fact('并不失眠，也没有畏寒。', fact('不失眠', '并不失眠'),
                            fact('畏寒', '没有畏寒', field='cold_heat', polarity='negated', id='f2'))
        self.assertEqual(a['status'], 'ok')
        fs = m.case_state['fact_view']['facts']
        self.assertTrue(any(f['field']=='sleep' and f['value']=='失眠' and f['polarity']=='negated' for f in fs))
        self.assertTrue(any(f['field']=='symptoms' and f['value']=='畏寒' and f['polarity']=='negated' for f in fs))

    def test_whitespace_evidence_is_literal_after_unique_recovery(self):
        m, a = self.run_fact('我28岁，寒热和平时一样。', fact('28', '我 28 岁', field='age'),
                            fact('和平时一样', field='cold_heat', id='f2'))
        self.assertEqual(a['status'], 'ok')
        self.assertEqual(a['raw_delta']['facts'][0]['evidence'], '我 28 岁')
        self.assertEqual(m.case_state['fact_ledger']['records'][0]['evidence'], '我28岁')

    def test_repeated_or_paraphrased_evidence_is_not_repaired(self):
        for text, evidence in [('我28岁，我28岁。', '我 28 岁'), ('我28岁。', '我二十八岁')]:
            with self.subTest(text=text):
                m, a = self.run_fact(text, fact('28', evidence, field='age'))
                self.assertEqual(a['status'], 'error')
                self.assertEqual(m.case_state['fact_ledger']['records'], [])

    def test_degree_alias_is_supported_but_no_change_duration_rejected(self):
        m, a = self.run_fact('腹痛很轻。', fact('腹痛'), fact('很轻', field='severity', id='f2', about='f1'))
        self.assertEqual(a['status'], 'ok')
        self.assertEqual(m.case_state['severity'], '轻度')
        _, a = self.run_fact('持续时间没改。', fact('没改', field='duration'), memory=m)
        self.assertEqual(a['status'], 'error')
        self.assertIn('attribute_status', a['attempts'][0]['error'])

    def test_conversion_does_not_fill_empty_facts(self):
        m, a = self.run_fact('失眠，心悸先搁置。')
        self.assertEqual(a['raw_delta']['facts'], [])
        self.assertEqual(a['converted_delta']['facts'], [])
        self.assertEqual(m.case_state['fact_view']['facts'], [])

    def test_duplicate_id_alias_redirects_duration_and_preserves_input(self):
        inp = build_input('多梦八天。', contract_version=V2)
        raw = delta(fact('多梦'), fact('多梦', field='sleep', id='f2'),
                    fact('八天', field='duration', id='f3', about='f2'))
        saved = deepcopy(raw)
        out, changes = normalize_representation(raw, inp)
        self.assertEqual(raw, saved)
        self.assertEqual(len(out['facts']), 2)
        self.assertEqual(out['facts'][1]['about'], 'f1')
        self.assertTrue(any(c['kind']=='duplicate_alias' for c in changes))
        position_delta(out, inp)

    def test_old_contract_still_rejects_sleep_duration(self):
        raw = delta(fact('多梦', field='sleep'), fact('八天', field='duration', id='f2', about='f1'))
        with self.assertRaises(ValueError):
            position_delta(raw, build_input('多梦八天。'))
        position_delta(raw, build_input('多梦八天。', contract_version=V2))

    def test_schema_does_not_force_vague_duration_into_numeric_units(self):
        from jsonschema import Draft202012Validator
        from llm.fact_contract_runtime import decoding_schema
        inp = build_input('咳嗽已经很久了。', contract_version=V2)
        raw = delta(fact('咳嗽'), fact('很久', field='duration', id='f2', about='f1'))
        Draft202012Validator(decoding_schema(inp)).validate(raw)
        position_delta(raw, inp)

    def test_invalid_degree_target_is_not_automatically_repaired(self):
        m, _ = self.run_fact('耳痛剧烈。', fact('耳痛'), fact('剧烈', field='severity', id='f2', about='f1'))
        _, a = self.run_fact('耳痛不是剧烈，是轻微。', fact('轻微', '耳痛不是剧烈，是轻微',
            field='severity', operation='correct', about='t1:f1', replaces='t1:f1'), memory=m)
        self.assertEqual(a['status'], 'error')
        self.assertEqual(statuses(m.case_state['fact_ledger'])['t1:f2'], 'active')

    def test_non_deferred_uncertainty_is_clarified_and_answer_refines(self):
        client = FixedClient(delta(fact('失眠', '说不准有没有失眠', field='sleep', polarity='uncertain')),
                             delta(fact('失眠', '没有失眠', field='sleep', polarity='negated')))
        extractor = SemanticExtractor(client, contract_version=V2, recovery_mode='guarded_v10', prompt_version='coverage_v3')
        with patch('agent.graph._run_guideline', return_value={'advice': []}):
            agent = create_agent(Mock(), ConversationMemory(), fact_extractor=extractor)
            answer = agent.run('说不准有没有失眠。')
            self.assertIn('你现在有失眠吗', answer)
            agent.run('没有失眠。')
        self.assertEqual(agent.memory.case_state['fact_clarifications'], [])
        self.assertEqual(statuses(agent.memory.case_state['fact_ledger'])['t1:f1'], 'superseded')

    def test_normal_state_asks_details_without_manufacturing_denials(self):
        m, a = self.run_fact('寒热和平时一样。', fact('和平时一样', field='cold_heat'))
        self.assertEqual(a['status'], 'ok')
        self.assertIn('发热或畏寒', m.case_state['fact_clarifications'][0]['question'])
        self.assertFalse(any(f['polarity']=='negated' for f in m.case_state['fact_view']['facts']))
        m, a = self.run_fact('没有发热，也没有畏寒。', fact('发热','没有发热',polarity='negated'),
                            fact('畏寒','没有畏寒',polarity='negated',id='f2'), memory=m)
        self.assertEqual(a['status'], 'ok')
        self.assertEqual(m.case_state['fact_clarifications'], [])

    def test_fact_context_preserves_ledger_and_attribute_dependencies(self):
        from memory.fact_context import select_facts
        facts = [dict(id=str(i), field='symptoms', value='咳嗽', subject='patient',
                      polarity='present', temporality='current', about=None) for i in range(40)]
        facts[0].update(field='red_flags',value='呼吸困难')
        facts[-1].update(field='duration',value='三天',about='1')
        saved = deepcopy(facts)
        chosen, trace = select_facts(facts, '2', max_facts=8, byte_budget=3000)
        ids = {f['id'] for f in chosen}
        self.assertIn('0', ids)
        self.assertIn('2', ids)
        self.assertIn('39', ids)
        self.assertIn('1', ids)
        self.assertEqual(facts, saved)
        self.assertEqual(trace['total'], 40)

    def test_deferred_uncertain_output_cannot_hide_known_positive(self):
        m, _ = self.run_fact('腹痛四天。',fact('腹痛'),fact('四天',field='duration',id='f2',about='f1'))
        m, a = self.run_fact('腹痛的事情以后再谈。',fact('腹痛','腹痛的事情以后再谈',polarity='uncertain'),memory=m)
        self.assertEqual(a['status'],'ok')
        self.assertEqual(m.case_state['symptoms'],['腹痛'])
        self.assertEqual(m.case_state['contradictions'],[])
        self.assertEqual(m.case_state['duration'],'四天')
        self.assertEqual(a['raw_delta']['facts'][0]['polarity'],'uncertain')
        self.assertTrue(any(c['kind']=='retain_known_deferred' for c in a['representation_changes']))


if __name__ == '__main__':
    unittest.main()
