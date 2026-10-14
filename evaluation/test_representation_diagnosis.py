from copy import deepcopy
import unittest

from memory.runtime_fact_projection import derive_identical_concept_fields


class ConceptProjectionTests(unittest.TestCase):
    def test_hypothetical_denied_and_other_subject_retain_scope(self):
        facts = [dict(id='f1', field='symptoms', value='呼吸困难', polarity='present',
                      temporality='hypothetical', subject='other', about=None),
                 dict(id='f2', field='red_flags', value='呼吸困难', polarity='negated',
                      temporality='current', subject='patient', about=None)]
        original = deepcopy(facts)
        output, trace = derive_identical_concept_fields(facts)
        self.assertEqual(facts, original)
        self.assertEqual(len(output), 4)
        self.assertEqual(output[2]['temporality'], 'hypothetical')
        self.assertEqual(output[2]['subject'], 'other')
        self.assertEqual(output[3]['polarity'], 'negated')
        self.assertFalse(trace['risk_input_changed'])

    def test_no_clinical_expansion_and_no_conflict_removal(self):
        facts = [dict(field='symptoms', value=v, polarity=p, temporality='current',
                      subject='patient', about=None) for v, p in
                 [('胸痛','present'),('黑便','present'),('呼吸困难','present'),('呼吸困难','negated')]]
        output, _ = derive_identical_concept_fields(facts)
        self.assertEqual(len(output), 6)
        derived = [f for f in output if f['field'] == 'red_flags']
        self.assertEqual({f['value'] for f in derived}, {'呼吸困难'})
        self.assertEqual({f['polarity'] for f in derived}, {'present','negated'})
        twice, _ = derive_identical_concept_fields(output)
        self.assertEqual(twice, output)
