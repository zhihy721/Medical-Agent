"""Exercise real Flask handlers with a deterministic blocking actor and durable archive."""
import importlib
import json
import os
import logging
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from memory.memory import ConversationMemory
from memory.rollout_store import RolloutSessionStore


class ChatRequestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.boot=tempfile.TemporaryDirectory()
        config=Path(cls.boot.name)/'config.env'
        config.write_text('LLM_PROVIDER=mock\nDATA_DIR='+cls.boot.name+'/data\nLOG_DIR='+cls.boot.name+'/logs\n',encoding='utf8')
        with patch.dict(os.environ,{'MEDICAL_AGENT_CONFIG_PATH':str(config)}):
            cls.web=importlib.import_module('app')
        cls.web.app.config['TESTING']=True

    @classmethod
    def tearDownClass(cls):
        for logger in [logging.getLogger(), *[x for x in logging.Logger.manager.loggerDict.values() if isinstance(x,logging.Logger)]]:
            for handler in list(logger.handlers):
                if str(getattr(handler,'baseFilename','')).startswith(cls.boot.name):
                    handler.close();logger.removeHandler(handler)
        cls.boot.cleanup()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.store=RolloutSessionStore(self.temp.name)
        memory=ConversationMemory(user_id='user',session_id='sid',session_store=self.store)
        class Actor:
            calls=0
            def __init__(self):self.memory=memory;self.release=threading.Event();self.started=threading.Event();self.fail=False
            def get_case_snapshot(self):return {'risk_level':'LOW'}
            def run_stream(self,text,user_coords=None):
                self.calls+=1;memory.add_user(text);self.started.set()
                if self.fail:raise RuntimeError('upstream unavailable')
                yield {'kind':'safety_notice','response':'测试中的首段'}
                self.release.wait(2)
                memory.add_assistant('已记录')
                yield {'kind':'final','response':'已记录'}
        self.actor=Actor();self.entry={'agent':self.actor,'lock':threading.Lock()}
        self.patches=[patch.object(self.web,'session_store',self.store),patch.object(self.web,'_get_agent',return_value=self.entry),
                      patch.object(self.web,'get_config_status',return_value={})]
        for p in self.patches:p.start();self.addCleanup(p.stop)
        self.client=self.web.app.test_client()
        with self.client.session_transaction() as s:s.update(user_id='user',session_id='sid')

    def test_duplicate_pending_then_completed_replay_and_payload_conflict(self):
        body={'message':'头痛','request_id':'test-1'}
        response=self.client.post('/chat/stream',json=body,buffered=False)
        self.assertEqual(json.loads(next(response.response))['kind'],'safety_notice')
        second=self.web.app.test_client()
        with second.session_transaction() as s:s.update(user_id='user',session_id='sid')
        self.assertEqual(second.post('/chat/stream',json=body).status_code,409)
        self.assertEqual(self.actor.calls,1)
        self.actor.release.set();list(response.response);response.close()
        replay=self.client.post('/chat/stream',json=body)
        self.assertTrue(json.loads(replay.data)['replayed']);self.assertEqual(self.actor.calls,1)
        self.assertEqual(len(self.actor.memory.get_archived_messages()),2)
        self.assertEqual(self.client.post('/chat/stream',json={**body,'message':'别的内容'}).status_code,409)
        self.assertEqual(self.client.get('/chat/requests/test-1').json['status'],'completed')
        with second.session_transaction() as s:s.update(session_id='other')
        self.assertEqual(second.get('/chat/requests/test-1').status_code,404)

    def test_transport_close_keeps_notice_and_never_reexecutes(self):
        body={'message':'头痛','request_id':'closed'}
        response=self.client.post('/chat/stream',json=body,buffered=False)
        next(response.response);response.close()
        self.assertFalse(self.entry['lock'].locked())
        self.assertEqual(self.client.get('/chat/requests/closed').json['status'],'interrupted')
        self.assertEqual(self.client.post('/chat/stream',json=body).status_code,409)
        self.assertEqual(self.actor.calls,1)

    def test_failure_is_recorded_and_same_id_never_calls_again(self):
        self.actor.fail=True
        body={'message':'头痛','request_id':'failed'}
        response=self.client.post('/chat/stream',json=body)
        self.assertEqual(json.loads(response.data)['kind'],'error')
        self.assertEqual(self.client.get('/chat/requests/failed').json['status'],'failed')
        self.assertEqual(self.client.post('/chat/stream',json=body).status_code,409)
        self.assertEqual(self.actor.calls,1)
        self.assertEqual(len(self.actor.memory.get_archived_messages()),1)

    def test_busy_session_cannot_lose_its_lock_on_ttl_or_lru_eviction(self):
        now=[0];cache=self.web.SessionCache(ttl_seconds=1,max_entries=1,clock=lambda:now[0])
        entry=cache.get_or_create('old',lambda:self.actor);entry['lock'].acquire()
        try:
            now[0]=10
            self.assertIs(cache.get_or_create('old',lambda:None),entry)
            with self.assertRaises(RuntimeError):cache.get_or_create('new',lambda:None)
            cache.remove('old');self.assertEqual(len(cache),1)
        finally:entry['lock'].release()
        self.assertIsNot(cache.get_or_create('new',lambda:self.actor),entry)


if __name__=='__main__':unittest.main()
