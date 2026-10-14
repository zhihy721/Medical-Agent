from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
import time
from desktop.ownership import WorkspaceLease,recover_owned_pending
from memory.rollout_store import RolloutSessionStore

class OwnershipTests(unittest.TestCase):
    def test_backend_lease_remains_exclusive_until_draining_process_exits(self):
        # Exercise main returning while a non-daemon task still drains. The
        # second process must not recover pending requests during this gap.
        with tempfile.TemporaryDirectory() as data:
            root=Path(data);release=root/'release';returned=root/'returned'
            script=r'''
import io,json,pathlib,sys,threading,time,types
from unittest.mock import patch
from desktop import backend
p=pathlib.Path(sys.argv[1])
class Server:
 server_port=12345
 def shutdown(self):pass
 def server_close(self):pass
 def serve_forever(self):
  def drain():
   while not (p/'release').exists():time.sleep(.02)
  threading.Thread(target=drain,daemon=False).start()
application=types.SimpleNamespace(session_store=None,shutdown_mcp_servers=lambda:None)
sys.stdin=io.StringIO(json.dumps(dict(token='x'*64))+'\nshutdown\n')
sys.argv=['backend','--project',str(pathlib.Path.cwd()),'--data',str(p),'--ready-file',str(p/'ready'),'--ui',str(p)]
with patch.object(backend,'configure',return_value=(pathlib.Path.cwd(),p)),patch.object(backend,'install',return_value=None),patch('desktop.ownership.recover_owned_pending',return_value=[]),patch('werkzeug.serving.make_server',return_value=Server()),patch.dict(sys.modules,{'app':application}):
 backend.main()
(p/'returned').write_text('main returned, worker still alive')
'''
            child=subprocess.Popen([sys.executable,'-X','utf8','-c',script,data],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            try:
                deadline=time.monotonic()+15
                while not returned.exists() and child.poll() is None and time.monotonic()<deadline:time.sleep(.03)
                self.assertTrue(returned.exists(),'backend fixture did not return')
                self.assertIsNone(child.poll())
                contender=None
                try:
                    with self.assertRaises(RuntimeError):contender=WorkspaceLease(data)
                finally:
                    if contender is not None:contender.close()
            finally:
                release.write_text('finish');child.communicate(timeout=10)
            self.assertEqual(child.returncode,0)
            with WorkspaceLease(data) as lease:self.assertTrue(lease.active)

    def test_live_owner_blocks_second_owner_then_releases(self):
        with tempfile.TemporaryDirectory() as data:
            a=WorkspaceLease(data)
            with self.assertRaises(RuntimeError):WorkspaceLease(data)
            a.close()
            with WorkspaceLease(data) as b:self.assertTrue(b.active)

    def test_pending_recovered_once_completed_reply_and_archive_unchanged(self):
        with tempfile.TemporaryDirectory() as data,WorkspaceLease(data) as lease:
            store=RolloutSessionStore(Path(data)/'data/sessions');store.register_session('s','u')
            store.claim_request('s','u','finished',{'message':'已完成'});store.append_message('s','user','已完成')
            store.finish_request('s','finished','completed',{'response':'旧答复'})
            store.claim_request('s','u','orphan',{'message':'发送中'});store.append_message('s','user','发送中')
            prefix=store.read_events('s');messages=store.get_messages('s')
            recovered=recover_owned_pending(store,lease)
            self.assertEqual(len(recovered),1);self.assertEqual(store.get_request('s','u','orphan')['status'],'interrupted')
            self.assertEqual(store.get_request('s','u','finished')['result'],{'response':'旧答复'})
            self.assertEqual(store.read_events('s')[:len(prefix)],prefix);self.assertEqual(store.get_messages('s'),messages)
            self.assertEqual(recover_owned_pending(store,lease),[])
            self.assertFalse(store.claim_request('s','u','orphan',{'message':'发送中'})['claimed'])

    def test_cannot_recover_unrelated_store_or_without_lease(self):
        with tempfile.TemporaryDirectory() as data,WorkspaceLease(data) as lease:
            store=RolloutSessionStore(Path(data)/'unrelated')
            with self.assertRaises(ValueError):recover_owned_pending(store,lease)
            with self.assertRaises(ValueError):recover_owned_pending(store,None)

    def test_rebuild_index_before_orphan_recovery(self):
        with tempfile.TemporaryDirectory() as data,WorkspaceLease(data) as lease:
            root=Path(data)/'data/sessions';store=RolloutSessionStore(root);store.register_session('s','u')
            store.claim_request('s','u','orphan',{'message':'x'});store.database.unlink()
            store=RolloutSessionStore(root);self.assertEqual(len(recover_owned_pending(store,lease)),1)
            self.assertEqual(store.get_request('s','u','orphan')['status'],'interrupted')

if __name__=='__main__':unittest.main()
