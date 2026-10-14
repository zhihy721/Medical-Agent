import unittest
from training.complaint_contract import build_complaints
from tools.risk_tool import risk_assessment


def symptom(value, polarity='present'):
    return {'field':'symptoms','value':value,'polarity':polarity}


class ComplaintTests(unittest.TestCase):
    def test_mention_order_does_not_confirm_or_rank(self):
        facts = [symptom('头痛'), symptom('咳嗽'), symptom('胸痛','negated')]
        a = build_complaints('头痛和咳嗽，没有胸痛。', facts)
        b = build_complaints('头痛和咳嗽，没有胸痛。', list(reversed(facts)))
        self.assertEqual(a,b)
        self.assertEqual(a['chief_complaints'],[])
        self.assertEqual(set(a['chief_complaint_candidates']),{'头痛','咳嗽'})

    def test_explicit_multiple_primary_problems_supported(self):
        text='我主要想问头痛和咳嗽。'
        confirmed=[{'value':v,'evidence':text,'start':0,'end':len(text)} for v in ['头痛','咳嗽']]
        result=build_complaints(text,[symptom('头痛'),symptom('咳嗽')],confirmed)
        self.assertEqual(len(result['chief_complaints']),2)
        self.assertEqual(result['chief_complaint_candidates'],[])

    def test_one_explicit_priority_leaves_other_candidate(self):
        text='主要想问头痛，咳嗽顺便提一下。'
        primary={'value':'头痛','evidence':'主要想问头痛','start':0,'end':6}
        result=build_complaints(text,[symptom('头痛'),symptom('咳嗽')],[primary])
        self.assertEqual(result['chief_complaint_candidates'],['咳嗽'])

    def test_invalid_evidence_and_nonpositive_primary_rejected(self):
        item={'value':'胸痛','evidence':'胸痛','start':0,'end':2}
        with self.assertRaises(ValueError):
            build_complaints('胸痛',[symptom('胸痛','negated')],[item])
        item['end']=1
        with self.assertRaises(ValueError):
            build_complaints('胸痛',[symptom('胸痛')],[item])

    def test_risk_uses_all_symptoms_not_chief_or_order(self):
        for complaints in [[], ['咳嗽'], ['胸痛','咳嗽'], ['咳嗽','胸痛']]:
            for facts in [['胸痛','咳嗽'], ['咳嗽','胸痛']]:
                state={'symptoms':facts,'past_history':['高血压'],
                       'chief_complaints':complaints,'chief_complaint':'咳嗽'}
                self.assertEqual(risk_assessment(state)['risk'],'HIGH')
        self.assertEqual(risk_assessment({'symptoms':['咳嗽'],'chief_complaints':[],
                                         'red_flags':['呼吸困难']})['risk'],'HIGH')


if __name__ == '__main__': unittest.main()
