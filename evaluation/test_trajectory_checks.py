import unittest
from evaluation.trajectory_checks import followup_limits


class RepeatCapTests(unittest.TestCase):
    def test_two_different_facts_can_be_clarified_twice(self):
        self.assertTrue(followup_limits(dict(followup_counts={'fact_confirmation':4},
                                            fact_confirmation_counts={'f1':2,'f2':2})))
        self.assertFalse(followup_limits(dict(fact_confirmation_counts={'f1':3})))

    def test_ordinary_slots_still_use_their_own_caps(self):
        self.assertFalse(followup_limits(dict(followup_counts={'sleep':2})))
        self.assertTrue(followup_limits(dict(problem_followup_counts={'f1':{'duration':2},'f2':{'duration':2}})))
