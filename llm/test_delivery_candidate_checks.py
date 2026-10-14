import unittest
from copy import deepcopy
from agent.test_semantic_runtime import delta
from training.test_fact_contract import fact
from llm.delivery_candidate_checks import error


class CandidateChecksTests(unittest.TestCase):
    def test_conversion_cannot_bypass_grounding(self):
        inp=dict(current_turn='体温正常。',known_facts=[],contract_version='semantic_facts_v2')
        self.assertIn('delivery_red_flag_without_concept_evidence',error(inp,delta(fact('呼吸困难','体温正常'))))

    def test_valid_input_is_read_only(self):
        inp=dict(current_turn='头疼。',known_facts=[],contract_version='semantic_facts_v2')
        pred=delta(fact('头疼','头疼'));original=deepcopy(pred)
        self.assertEqual(error(inp,pred),'')
        self.assertEqual(pred,original)

    def test_discarded_bad_resolution_is_rejected(self):
        inp=dict(current_turn='刚才确实呼吸困难，现在缓解了。',known_facts=[],contract_version='semantic_facts_v2')
        pred=delta(fact('呼吸困难','刚才确实呼吸困难',field='red_flags'),
                   fact('呼吸困难','现在缓解了',field='red_flags',id='f2'))
        # The missing raw concept is also a valid grounding rejection. This
        # delta must never be accepted just because aliases discard one fact.
        self.assertTrue(error(inp,pred))

if __name__=='__main__':unittest.main()
