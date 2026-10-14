"""Isolated Flask integration with private config/data and no model calls."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class ArchiveBackendTests(unittest.TestCase):
    def test_status_replays_full_archive_after_cache_eviction_and_reset_keeps_old(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp)/'config.env'
            config.write_text('LLM_PROVIDER=mock\nDATA_DIR='+str(Path(temp)/'data')+'\nLOG_DIR='+str(Path(temp)/'logs')+'\n',encoding='utf8')
            code = '''
import json
import app as a
client=a.app.test_client()
assert client.get('/status').status_code==200
with client.session_transaction() as s:
 sid=s['session_id'];uid=s['user_id']
m=a.session_cache.get_or_create(sid,lambda:None)['agent'].memory
for i in range(20):
 m.add_user('question-'+str(i));m.add_assistant('answer-'+str(i))
assert len(m.history)==12
assert len(client.get('/status').get_json()['messages'])==40
a.session_cache.remove(sid)
messages=client.get('/status').get_json()['messages']
assert len(messages)==40 and messages[0]['content']=='question-0'
assert client.post('/reset').get_json()['messages']==[]
assert len(a.session_store.get_messages(sid))==40
assert any(r['session_id']==sid for r in a.session_store.list_sessions(uid))
assert a.session_store.database.exists()
print(json.dumps({'archive_messages':40,'context_tail_messages':12,'reset_preserves_old':True,'backend':'jsonl_sqlite'}))
'''
            env = {k:v for k,v in os.environ.items() if not k.startswith('AGENT_')}
            env['MEDICAL_AGENT_CONFIG_PATH'] = str(config)
            result = subprocess.run([os.sys.executable,'-X','utf8','-c',code],cwd=Path(__file__).resolve().parents[1],
                                    env=env,capture_output=True,text=True,encoding='utf8',timeout=45)
            self.assertEqual(result.returncode,0,result.stderr[-2500:])
            self.assertIn('"archive_messages": 40',result.stdout)


if __name__=='__main__':
    unittest.main()
