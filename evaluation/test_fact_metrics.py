import copy
import unittest

from evaluation.fact_metrics import score_delta, summarize
from llm.fact_contract import build_input


def fact(id, field, value, evidence, about=None, **changes):
    result = dict(id=id, field=field, value=value, evidence=evidence, about=about,
                  polarity='present', subject='patient', temporality='current',
                  operation='assert', replaces=None)
    return dict(result, **changes)


class FactMetricsTests(unittest.TestCase):
    def setUp(self):
        self.inp = build_input('咳嗽三天，头痛一天，没有胸痛。')
        self.gold = {'facts': [fact('f1', 'symptoms', '咳嗽', '咳嗽'),
                               fact('f2', 'symptoms', '头痛', '头痛'),
                               fact('f3', 'duration', '三天', '三天', 'f1'),
                               fact('f4', 'duration', '一天', '一天', 'f2'),
                               fact('f5', 'symptoms', '胸痛', '没有胸痛', polarity='negated')]}

    def test_alpha_renaming_and_order(self):
        pred = copy.deepcopy(self.gold)
        ids = {f'f{i}': f'f{i+10}' for i in range(1, 6)}
        for f in pred['facts']:
            f['id'] = ids[f['id']]
            f['about'] = ids.get(f['about'])
        pred['facts'].reverse()
        self.assertTrue(score_delta(pred, self.gold, self.inp)['strict_success'])

    def test_swapped_duration_is_semantic_match_but_wrong_binding(self):
        pred = copy.deepcopy(self.gold)
        pred['facts'][2]['about'], pred['facts'][3]['about'] = 'f2', 'f1'
        score = score_delta(pred, self.gold, self.inp)
        self.assertTrue(score['semantics_exact'])
        self.assertFalse(score['bindings_exact'])

    def test_polarity_and_subject_errors(self):
        for field, value in [('polarity', 'present'), ('subject', 'other')]:
            pred = copy.deepcopy(self.gold)
            pred['facts'][-1][field] = value
            score = score_delta(pred, self.gold, self.inp)
            self.assertTrue(score['contract_valid'])
            self.assertFalse(score['semantics_exact'])

    def test_literal_span_without_negation_is_not_grounded_gold(self):
        pred = copy.deepcopy(self.gold)
        pred['facts'][-1]['evidence'] = '胸痛'
        score = score_delta(pred, self.gold, self.inp)
        self.assertTrue(score['operations_exact'])
        self.assertFalse(score['evidence_exact'])

    def test_explicit_evidence_alternative_only(self):
        pred = copy.deepcopy(self.gold)
        pred['facts'][0]['evidence'] = '咳嗽三天'
        self.assertFalse(score_delta(pred, self.gold, self.inp)['strict_success'])
        self.assertTrue(score_delta(pred, self.gold, self.inp, [pred])['strict_success'])
        pred['facts'][-1]['polarity'] = 'present'
        with self.assertRaises(ValueError):
            score_delta(self.gold, self.gold, self.inp, [pred])

    def test_missing_extra_duplicate_and_malformed_are_failures(self):
        missing = copy.deepcopy(self.gold)
        missing['facts'].pop()
        duplicate = copy.deepcopy(self.gold)
        duplicate['facts'].append(dict(duplicate['facts'][-1], id='f6'))
        scores = [score_delta(p, self.gold, self.inp) for p in [missing, duplicate, '{', None]]
        self.assertEqual(summarize(scores)['attempts'], 4)
        self.assertEqual(summarize(scores)['passed']['strict_success'], 0)
        self.assertEqual(scores[0]['semantic_counts']['matched'], 4)

    def test_invalid_gold_raises(self):
        gold = copy.deepcopy(self.gold)
        gold['facts'][2]['about'] = 'absent'
        with self.assertRaises(ValueError):
            score_delta(self.gold, gold, self.inp)

    def test_empty_is_success_only_against_empty(self):
        self.assertTrue(score_delta({'facts': []}, {'facts': []}, self.inp)['strict_success'])
        self.assertFalse(score_delta({'facts': []}, self.gold, self.inp)['strict_success'])

    def test_correct_versus_assert_and_wrong_previous_record(self):
        known = [dict(id=f't1:f{i}', field='symptoms', value=value, polarity='present',
                      subject='patient', temporality='current', about=None,
                      source='current_dialogue', source_turn=1, status='active')
                 for i, value in [(1, '咳嗽'), (2, '头痛')]]
        inp = build_input('刚才说错了，没有咳嗽。', 2, known)
        gold = {'facts': [fact('f1', 'symptoms', '咳嗽', inp['current_turn'],
                               polarity='negated', operation='correct', replaces='t1:f1')]}
        for changes in [dict(operation='assert', replaces=None), dict(replaces='t1:f2')]:
            pred = {'facts': [dict(gold['facts'][0], **changes)]}
            score = score_delta(pred, gold, inp)
            self.assertTrue(score['semantics_exact'])
            self.assertFalse(score['operations_exact'])


if __name__ == '__main__':
    unittest.main()
