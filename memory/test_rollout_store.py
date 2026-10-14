"""Durability/recovery and archive/context separation through public interfaces."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from memory.context_window import build_context
from memory.memory import ConversationMemory
from memory.rollout_store import RolloutSessionStore


class RolloutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = RolloutSessionStore(self.root)

    def memory(self, store=None):
        return ConversationMemory(session_store=store or self.store, session_id='session1', user_id='user1')

    def test_full_archive_survives_trim_restart_and_summary_changes(self):
        m = self.memory()
        for i in range(20):
            m.add_user(f'问题{i}')
            m.add_assistant(f'回答{i}')
        self.assertEqual(len(m.history), 12)
        self.assertEqual(len(m.get_archived_messages()), 40)
        reopened = self.memory(RolloutSessionStore(self.root))
        self.assertEqual(reopened.history, m.history)
        self.assertEqual(reopened.get_archived_messages()[0]['content'], '问题0')
        text = reopened.get_prompt_context_text()
        self.assertIn('问题19', text)
        self.assertNotIn('问题0', text)

    def test_sqlite_rebuilt_from_rollout_and_amendment_is_preserved(self):
        m = self.memory()
        m.add_user('耳痛。')
        m.add_assistant('建议。')
        m.amend_last_assistant('建议。', '建议。\n医院结果。')
        m.case_state['city'] = '上海'
        m._persist_session_state()
        self.store.database.unlink()
        fresh = RolloutSessionStore(self.root)
        self.assertEqual(fresh.list_sessions('user1')[0]['message_count'], 2)
        restored = self.memory(fresh)
        self.assertEqual(restored.case_state['city'], '上海')
        self.assertEqual(restored.get_archived_messages()[-1]['content'], '建议。\n医院结果。')
        events = fresh.read_events('session1')
        self.assertEqual(next(e for e in events if e['type']=='message' and e['role']=='assistant')['content'], '建议。')

    def test_unindexed_complete_event_recovered_incomplete_tail_removed(self):
        m = self.memory()
        m.add_user('第一条')
        events = self.store.read_events('session1')
        event = dict(events[-1], seq=events[-1]['seq']+1, type='message', role='user', content='写入后崩溃')
        with self.store._path('session1').open('ab') as stream:
            stream.write((json.dumps(event, ensure_ascii=False)+'\n').encode('utf8'))
            stream.write(b'{"incomplete":')
        fresh = RolloutSessionStore(self.root)
        self.assertEqual(fresh.get_messages('session1')[-1]['content'], '写入后崩溃')
        fresh.append_message('session1', 'assistant', '恢复后回答')
        self.assertEqual(fresh.get_messages('session1')[-1]['content'], '恢复后回答')

    def test_malformed_complete_event_fails_without_silent_loss(self):
        self.store.register_session('s', 'u')
        with self.store._path('s').open('ab') as stream:
            stream.write(b'not-json\n')
        with self.assertRaises(json.JSONDecodeError):
            self.store.get_messages('s')

    def test_legacy_import_preserves_file_and_marks_incomplete_archive(self):
        legacy = dict(history=[['user','旧问题'],['assistant','旧回答']], case_state={'city':'上海'})
        path = self.root / 'session1.json'
        path.write_text(json.dumps(legacy, ensure_ascii=False), encoding='utf8')
        self.store.register_session('session1', 'user1')
        self.store.register_session('session1', 'user1')
        self.assertEqual(len(self.store.get_messages('session1')), 2)
        self.assertEqual(json.loads(path.read_text(encoding='utf8')), legacy)
        event = next(e for e in self.store.read_events('session1') if e['type']=='legacy_import')
        self.assertEqual(event['archive_completeness'], 'retained_tail_only')

    def test_concurrent_writers_have_contiguous_sequence(self):
        self.store.register_session('s', 'u')
        def append(i):
            RolloutSessionStore(self.root).append_message('s', 'user', str(i))
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(append, range(20)))
        events = self.store.read_events('s')
        self.assertEqual([e['seq'] for e in events], list(range(1, len(events)+1)))
        self.assertEqual(len(self.store.get_messages('s')), 20)

    def test_interrupted_legacy_import_resumes_without_loss_or_duplication(self):
        from unittest.mock import patch
        legacy = dict(history=[['user', '旧问题'], ['assistant', '旧回答']], case_state={'city': '上海'})
        (self.root/'session1.json').write_text(json.dumps(legacy, ensure_ascii=False), encoding='utf8')
        append = self.store._append
        def crash(db, sid, payload):
            result = append(db, sid, payload)
            if payload.get('source') == 'legacy_retained_tail':
                raise RuntimeError('simulated process interruption after durable append')
            return result
        with patch.object(self.store, '_append', side_effect=crash):
            with self.assertRaises(RuntimeError):
                self.store.register_session('session1', 'user1')
        fresh = RolloutSessionStore(self.root)
        fresh.register_session('session1', 'user1')
        fresh.register_session('session1', 'user1')
        self.assertEqual([m['content'] for m in fresh.get_messages('session1')], ['旧问题', '旧回答'])
        self.assertEqual(fresh.get_case_state('session1', {}), legacy['case_state'])

    def test_owner_and_path_isolation(self):
        self.store.register_session('s', 'u')
        with self.assertRaises(ValueError):
            self.store.register_session('s', 'other')
        self.assertEqual(self.store.list_sessions('other'), [])
        with self.assertRaises(ValueError):
            self.store.append_message('../outside', 'user', 'x')

    def test_oversize_context_does_not_modify_archive(self):
        m = self.memory()
        m.add_user('长'*1000)
        before = self.store._path('session1').read_bytes()
        with self.assertRaises(ValueError):
            m.get_prompt_context_text()
        self.assertEqual(before, self.store._path('session1').read_bytes())

    def test_context_selects_whole_messages_and_respects_budget(self):
        result = build_context('摘要'*100, [('user','甲'*10), ('assistant','乙'*10), ('user','丙'*10)], token_budget=80)
        self.assertLessEqual(result['estimated_tokens'], 80)
        self.assertIn('丙'*10, result['text'])
        self.assertFalse(result['summary_included'])
        self.assertEqual(result['retained_messages'], 1)

    def test_delta_archive_limits_growth_and_recovers_after_index_deletion(self):
        self.store.register_session('s', 'u')
        state = {'large_unchanged': '证据'*20000, 'count': 0, 'events': []}
        self.store.save_snapshot('s', [], state)
        for i in range(1, 41):
            self.store.append_message('s', 'user', f'问题{i}')
            state['count'] = i
            state['events'].append({'turn': i})
            self.store.save_snapshot('s', [['user', f'问题{i}']], state)
        events = self.store.read_events('s')
        full = [e['seq'] for e in events if e['type'] == 'checkpoint']
        patches = [e for e in events if e['type'] == 'checkpoint_patch']
        self.assertGreaterEqual(len(full), 3)
        self.assertGreater(len(patches), 30)
        self.assertTrue(all(b-a <= 35 for a,b in zip(full, full[1:])))
        self.assertLess(self.store._path('s').stat().st_size, 800000)
        self.store.database.unlink()
        fresh = RolloutSessionStore(self.root)
        self.assertEqual(fresh.get_case_state('s', {}), state)
        self.assertEqual(fresh.get_history('s'), [('user', '问题40')])
        self.assertEqual(len(fresh.get_messages('s')), 40)

    def test_durable_delta_recovered_after_index_transaction_interruption(self):
        from unittest.mock import patch
        self.store.register_session('s', 'u')
        self.store.save_snapshot('s', [], {'large': '原文'*1000, 'count': 0})
        append = self.store._append
        def crash(db, sid, payload):
            result = append(db, sid, payload)
            if payload['type'] == 'checkpoint_patch':
                raise RuntimeError('crash after JSONL append before SQLite commit')
            return result
        with patch.object(self.store, '_append', side_effect=crash):
            with self.assertRaises(RuntimeError):
                self.store.save_snapshot('s', [], {'large': '原文'*1000, 'count': 1})
        self.assertEqual(RolloutSessionStore(self.root).get_case_state('s', {})['count'], 1)

    def test_complete_delta_with_wrong_checksum_fails_without_discard(self):
        for checksum in ['base_sha256', 'result_sha256']:
            sid = checksum
            self.store.register_session(sid, 'u')
            self.store.save_snapshot(sid, [], {'large': '原文'*1000, 'count': 0})
            self.store.save_snapshot(sid, [], {'large': '原文'*1000, 'count': 1})
            path = self.store._path(sid)
            events = self.store.read_events(sid)
            self.assertEqual(events[-1]['type'], 'checkpoint_patch')
            events[-1][checksum] = '0'*64
            path.write_text(''.join(json.dumps(e, ensure_ascii=False)+'\n' for e in events), encoding='utf8')
            before = path.read_bytes()
            self.store.database.unlink()
            fresh = RolloutSessionStore(self.root)
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                fresh.get_case_state(sid, {})
            self.assertEqual(path.read_bytes(), before)

    def test_old_sqlite_index_migrates_without_changing_old_events(self):
        self.store.register_session('s', 'u')
        self.store.save_snapshot('s', [], {'city': '上海'})
        before = self.store._path('s').read_bytes()
        db = sqlite3.connect(self.store.database)
        try:
            with db:
                db.execute('ALTER TABLE threads DROP COLUMN checkpoint_seq')
        finally:
            db.close()
        fresh = RolloutSessionStore(self.root)
        self.assertEqual(fresh.list_sessions('u')[0]['session_id'], 's')
        self.assertEqual(fresh.get_case_state('s', {}), {'city': '上海'})
        self.assertEqual(self.store._path('s').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
