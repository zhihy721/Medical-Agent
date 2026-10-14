import json
import tempfile
import unittest
from memory.rollout_store import RolloutSessionStore


class RequestLifecycleTests(unittest.TestCase):
    def test_durable_claim_replay_conflict_and_index_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RolloutSessionStore(tmp)
            store.register_session('session', 'owner')
            self.assertTrue(store.claim_request('session', 'owner', 'req-1', {'message': '头痛'})['claimed'])
            self.assertFalse(store.claim_request('session', 'owner', 'req-1', {'message': '头痛'})['claimed'])
            with self.assertRaises(ValueError):
                store.claim_request('session', 'owner', 'req-1', {'message': '不同内容'})
            with self.assertRaises(ValueError):
                store.get_request('session', 'other', 'req-1')
            store.finish_request('session', 'req-1', 'completed', {'response': '已记录'})
            new = RolloutSessionStore(tmp)
            record = new.claim_request('session', 'owner', 'req-1', {'message': '头痛'})
            self.assertFalse(record['claimed'])
            self.assertEqual(record['result']['response'], '已记录')
            with new._connection() as db:
                db.execute('DELETE FROM requests')
                db.execute("UPDATE threads SET seq=0,offset=0,message_count=0,snapshot='{}',title='',checkpoint_seq=0")
            self.assertEqual(new.get_request('session', 'owner', 'req-1')['status'], 'completed')
            self.assertEqual([e['type'] for e in new.read_events('session')],
                             ['session_metadata', 'request_started', 'request_finished'])

    def test_unfinished_request_is_never_automatically_claimed_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RolloutSessionStore(tmp)
            store.register_session('session', 'owner')
            store.claim_request('session', 'owner', 'req', {'message': '头痛'})
            new = RolloutSessionStore(tmp)
            self.assertFalse(new.claim_request('session', 'owner', 'req', {'message': '头痛'})['claimed'])
            self.assertEqual(new.get_request('session', 'owner', 'req')['status'], 'pending')


if __name__ == '__main__':
    unittest.main()
