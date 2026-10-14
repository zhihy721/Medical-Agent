"""Focused boundary/fault tests, without model calls or patient data."""
from copy import deepcopy
import errno
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from agent.input_safety_signal import classify
from memory.fact_context import select_facts
from memory.rollout_store import RolloutSessionStore

class AcceptanceBoundaries(unittest.TestCase):
    def test_current_subject_and_non_current_matrix(self):
        cases={
            '爸爸今天呼吸困难。他喘不过气；我只有鼻塞。':False,
            '我姐姐现在无法呼吸，我本人只有鼻塞。':False,
            '我没有呼吸困难，也没有胸痛。':False,
            '如果以后我呼吸困难再说。':False,
            '资料中写“我现在无法呼吸”，不是我的描述。':False,
            '我本人现在呼吸困难，已经三分钟。':True,
            '我爸爸没有呼吸困难，我本人现在无法呼吸。':True,
        }
        for text,expected in cases.items():
            with self.subTest(text=text):self.assertEqual(bool(classify(text)['positive']),expected)

    def test_fact_compaction_keeps_safety_parent_and_archive(self):
        facts=[dict(id=f'f{i}',field='symptoms',value=f'记录{i}',subject='patient',polarity='present',temporality='current',about=None) for i in range(45)]
        facts[0].update(field='red_flags',value='呼吸困难')
        facts.append(dict(id='duration',field='duration',value='五天',subject='patient',polarity='present',temporality='current',about='f44'))
        before=deepcopy(facts);output,audit=select_facts(facts,'f44')
        self.assertLessEqual(len(output),32);self.assertIn('f0',[f['id'] for f in output])
        self.assertIn('duration',[f['id'] for f in output]);self.assertIn('f44',[f['id'] for f in output])
        self.assertEqual(facts,before);self.assertTrue(audit['omitted_ids']);self.assertFalse(audit['archive_modified'])
        self.assertLessEqual(len(json.dumps(output,ensure_ascii=False,separators=(',',':')).encode()),6500)

    def test_required_overflow_is_explicit_and_does_not_mutate(self):
        facts=[dict(id=f'f{i}',field='red_flags',value='安全事实'+str(i),subject='patient',polarity='present',temporality='current',about=None) for i in range(33)]
        before=deepcopy(facts)
        with self.assertRaisesRegex(ValueError,'required fact context exceeds'):select_facts(facts)
        self.assertEqual(facts,before)

    def test_rollout_open_failure_preserves_committed_prefix(self):
        with tempfile.TemporaryDirectory() as root:
            store=RolloutSessionStore(root);store.register_session('s','u');store.append_message('s','user','已提交消息')
            path=store._path('s');before=path.read_bytes();original=Path.open
            def fail(target,*args,**kwargs):
                if target==path and args and args[0]=='ab':raise OSError(errno.ENOSPC,'controlled disk full')
                return original(target,*args,**kwargs)
            with patch.object(Path,'open',fail):
                with self.assertRaises(OSError):store.append_message('s','assistant','不可写消息')
            self.assertEqual(path.read_bytes(),before)
            fresh=RolloutSessionStore(root);self.assertEqual([m['content'] for m in fresh.get_messages('s')],['已提交消息'])
            fresh.append_message('s','assistant','恢复可写');self.assertEqual(len(fresh.get_messages('s')),2)

    def test_fsync_failure_is_reported_and_next_access_reconciles_once(self):
        with tempfile.TemporaryDirectory() as root:
            store=RolloutSessionStore(root);store.register_session('s','u');store.append_message('s','user','原消息')
            with patch('memory.rollout_store.os.fsync',side_effect=OSError(errno.EIO,'controlled fsync failure')):
                with self.assertRaises(OSError):store.append_message('s','assistant','写出但持久性未确认')
            # A complete line MAY survive an I/O error; never claim rollback or
            # silently repeat. Recovery must reconcile authoritative JSONL once.
            fresh=RolloutSessionStore(root);messages=fresh.get_messages('s')
            self.assertEqual([m['content'] for m in messages],['原消息','写出但持久性未确认'])
            self.assertEqual(fresh.get_messages('s'),messages)

if __name__=='__main__':unittest.main()
