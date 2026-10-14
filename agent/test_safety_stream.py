import json
import threading
import unittest
from unittest.mock import patch
from flask import Flask
from agent import test_semantic_runtime as runtime_tests
from agent.test_semantic_runtime import delta
from training.test_fact_contract import fact
from agent.safety_stream import stream_response


class SafetyStreamTests(unittest.TestCase):
    agent=runtime_tests.SemanticRuntimeTests.agent

    def setUp(self):
        runtime_tests.SemanticRuntimeTests.setUp(self)
        self.hospital.stop()  # These tests exercise the real hospital boundary.

    def high(self):
        agent,_=self.agent(delta(fact('呼吸困难',field='red_flags')),delta())
        agent.memory.update_tool_location('上海')
        return agent

    def test_first_fragment_precedes_hospital_call_and_final_keeps_one_assistant_message(self):
        agent=self.high();calls=[]
        def tool(**kwargs):
            calls.append(kwargs);return {'status':'error'}
        with patch('tools.registry.default_registry.get',return_value=tool):
            stream=agent.run_stream('呼吸困难。')
            notice=next(stream)
            self.assertEqual(notice['kind'],'safety_notice');self.assertIn('需要优先警惕',notice['response'])
            self.assertEqual(calls,[])
            final=next(stream);self.assertEqual(final['kind'],'final')
            self.assertIn('不要等待查询结果',final['response']);self.assertEqual(len(calls),1)
            with self.assertRaises(StopIteration):next(stream)
            self.assertEqual([t for role,t in agent.memory.history if role=='assistant'],[final['response']])
            # Next synchronous turn resets the checkpoint's defer flag.
            response=agent.run('还是这样。');self.assertIn('医院查询暂不可用',response)
            self.assertEqual(len(calls),2)

    def test_disconnect_after_notice_does_not_start_tool_or_leave_background_work(self):
        agent=self.high()
        with patch('tools.registry.default_registry.get') as get:
            stream=agent.run_stream('呼吸困难。');notice=next(stream);stream.close()
            get.assert_not_called()
        self.assertEqual(agent.memory.history[-1],('assistant',notice['response']))

    def test_http_first_chunk_before_tool_and_session_lock_released_on_close(self):
        agent=self.high();lock=threading.Lock();app=Flask(__name__)
        @app.post('/test')
        def route():return stream_response({'agent':agent,'lock':lock},'呼吸困难。',None,
            lambda a,r:{'response':r,'messages':a.memory.history})
        with patch('tools.registry.default_registry.get') as get,app.test_client() as client:
            response=client.post('/test',buffered=False)
            chunk=json.loads(next(response.response));self.assertEqual(chunk['kind'],'safety_notice')
            self.assertTrue(lock.locked());get.assert_not_called()
            response.close();self.assertFalse(lock.locked());get.assert_not_called()

    def test_low_has_only_final_event(self):
        agent,_=self.agent(delta(fact('头痛')))
        events=list(agent.run_stream('头痛。'))
        self.assertEqual([e['kind'] for e in events],['final'])


if __name__=='__main__':unittest.main()
