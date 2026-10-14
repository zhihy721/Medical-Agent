import unittest
from training.formal_schedule import data_signature, resume_windows, schedule


class FormalScheduleTests(unittest.TestCase):
    def setUp(self):
        self.rows=[{'sample_id':f's{i}', 'source_family_id':f'f{i//2}', 'split':'train'}
                   for i in range(19)]

    def test_shuffle_resume_and_tail_without_loss_or_duplicate(self):
        plan=schedule(self.rows,seed=17,epochs=2,accumulation=8)
        self.assertEqual(plan,schedule(self.rows,seed=17,epochs=2,accumulation=8))
        self.assertEqual(plan['expected_optimizer_steps'],6)
        self.assertEqual([w['loss_divisor'] for w in plan['windows']],[8,8,3,8,8,3])
        suffix=resume_windows(plan,2,data_signature_value=data_signature(self.rows))
        self.assertEqual(suffix['samples_committed'],16)
        self.assertEqual(plan['windows'][2:],suffix['next_windows'])
        for epoch in (0,1):
            ids=[sid for w in plan['windows'] if w['epoch']==epoch for sid in w['sample_ids']]
            self.assertCountEqual(ids,[r['sample_id'] for r in self.rows])

    def test_validation_and_stale_resume_rejected(self):
        with self.assertRaises(ValueError):
            schedule(self.rows+[{'sample_id':'v','source_family_id':'fv','split':'validation'}])
        plan=schedule(self.rows)
        with self.assertRaisesRegex(ValueError,'data mismatch'):
            resume_windows(plan,1,data_signature_value='wrong')
        changed=[dict(r) for r in self.rows]
        changed[0]['target']={'facts':[]}
        self.assertNotEqual(data_signature(changed),data_signature(self.rows))
        with self.assertRaises(ValueError):
            resume_windows(plan,-1,data_signature_value=data_signature(self.rows))
