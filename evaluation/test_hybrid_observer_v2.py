from copy import deepcopy
import unittest
from evaluation.hybrid_observer_v2 import observe_trajectory_row


class ObserverTests(unittest.TestCase):
    def test_current_confirmation_key_and_legacy_key_without_mutating_actor_row(self):
        for confirmation in ({'candidate_concept':'心悸'},{'value':'心悸'}):
            row={'case_state':{},'plan':{'fact_confirmation':confirmation}}
            original=deepcopy(row)
            self.assertEqual(observe_trajectory_row(row)['confirmation_value'],'心悸')
            self.assertEqual(row,original)


if __name__=='__main__':unittest.main()
