import unittest
from jsonschema import Draft202012Validator
from agent.complaint_intent import ComplaintIntent,intent_schema
from agent.test_intent_fact_seam import selected_memory
from agent.test_semantic_runtime import FixedClient


class IntentSchemaTests(unittest.TestCase):
    def test_schema_allows_either_decision_but_rejects_invented_id_and_ellipsis(self):
        inp=dict(current_turn='鼻塞和乏力都想重点了解。',problems=[dict(id='t1:f1'),dict(id='t1:f3')])
        validator=Draft202012Validator(intent_schema(inp))
        event=dict(problem_id='t1:f1',operation='select',evidence=inp['current_turn'])
        self.assertTrue(validator.is_valid({'events':[]}))
        self.assertTrue(validator.is_valid({'events':[event]}))
        self.assertTrue(validator.is_valid({'events':[{**event,'operation':'withdraw'}]}))
        self.assertFalse(validator.is_valid({'events':[{**event,'problem_id':'unknown'}]}))
        self.assertFalse(validator.is_valid({'events':[{**event,'evidence':'鼻塞...重点了解'}]}))

    def test_empty_problems_schema_forces_no_events(self):
        validator=Draft202012Validator(intent_schema(dict(current_turn='没有别的了。',problems=[])))
        self.assertTrue(validator.is_valid({'events':[]}));self.assertFalse(validator.is_valid({'events':[{}]}))

    def test_real_intent_seam_requests_schema_only_when_opted_in(self):
        class Capturing(FixedClient):
            def call(self,messages,**kwargs):self.kwargs=kwargs;return super().call(messages,**kwargs)
        for mode in ['off','whole_turn_v1']:
            m=selected_memory();client=Capturing({'events':[]})
            audit=ComplaintIntent(client,structured_mode=mode).run(m,'没有别的了。')
            self.assertEqual(audit['status'],'ok');self.assertEqual('json_schema' in client.kwargs,mode!='off')


if __name__=='__main__':unittest.main()
