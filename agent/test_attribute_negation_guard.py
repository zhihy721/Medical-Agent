"""Scope alarms preserve evidence; fixed clients do not prove model recovery."""
import unittest
from agent.semantic_extraction import SemanticExtractor
from agent.test_semantic_runtime import FixedClient, delta
from memory.memory import ConversationMemory
from training.test_fact_contract import fact
from llm.runtime_extraction_guard import check_attribute_negation, classify


class AttributeNegationTests(unittest.TestCase):
    def test_rejected_denial_is_repaired_by_model_without_deleting_old_problem(self):
        memory = ConversationMemory(); memory.enable_fact_mode()
        text = '喉咙疼还是四天，不是耳朵的时间。'
        first = delta(fact('耳痛'), fact('三天', id='f2', field='duration', about='f1'))
        wrong = delta(fact('耳痛', '不是耳朵的时间', polarity='negated'))
        fixed = delta(fact('咽痛', '喉咙疼还是四天'),
                      fact('四天', '喉咙疼还是四天', id='f2', field='duration', about='f1'))
        client = FixedClient(first, wrong, fixed)
        extractor = SemanticExtractor(client, recovery_mode='guarded_v4')
        extractor.run(memory, '耳痛三天。')
        result = extractor.run(memory, text)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(len(result['attempts']), 2)
        self.assertEqual(result['attempts'][0]['raw_delta'], wrong)
        self.assertEqual(result['attempts'][0]['error'], 'attribute_ownership_is_not_symptom_evidence')
        self.assertIn('耳痛', memory.get_case_state()['symptoms'])
        self.assertEqual(len(memory.get_case_state()['fact_ledger']['turns']), 2)

    def test_failed_second_attempt_preserves_state_and_explicit_error(self):
        memory = ConversationMemory(); memory.enable_fact_mode()
        wrong = delta(fact('耳痛', '不是耳朵的时间', polarity='negated'))
        extractor = SemanticExtractor(FixedClient(delta(fact('耳痛')), wrong, wrong), recovery_mode='guarded_v4')
        extractor.run(memory, '耳痛。')
        result = extractor.run(memory, '不是耳朵的时间。')
        self.assertEqual(result['status'], 'error')
        self.assertIn('耳痛', memory.get_case_state()['symptoms'])
        self.assertEqual(len(result['attempts']), 2)

    def test_actual_symptom_denial_and_other_target_are_not_rejected(self):
        for value, evidence in [('耳痛','我没有耳痛'), ('耳痛','不是耳朵的时间，我没有耳痛'),
                                ('耳痛','不是喉咙的时间')]:
            check_attribute_negation({'known_facts': [], 'current_turn': evidence},
                delta(fact(value, evidence, polarity='negated')), classify(evidence))

    def test_legacy_v3_is_unchanged(self):
        memory = ConversationMemory();memory.enable_fact_mode()
        client = FixedClient(delta(fact('耳痛', '不是耳朵的时间', polarity='negated')))
        result = SemanticExtractor(client, recovery_mode='guarded_v3').run(memory, '不是耳朵的时间。')
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(len(client.calls), 1)

    def test_attribute_only_evidence_cannot_reassert_a_symptom_either(self):
        for polarity in ['present','negated','uncertain']:
            text='不是耳朵的时间'
            with self.subTest(polarity=polarity),self.assertRaisesRegex(ValueError,'attribute_ownership_is_not_symptom_evidence'):
                check_attribute_negation({'known_facts':[],'current_turn':text},
                    delta(fact('耳痛',text,polarity=polarity)),classify(text))
        text='耳痛还在，不是耳朵的时间'
        check_attribute_negation({'known_facts':[],'current_turn':text},
            delta(fact('耳痛',text)),classify(text))


if __name__ == '__main__':
    unittest.main()
