"""Local session archive: durable JSONL events, rebuildable SQLite state/index.

SQLite serializes writers across processes. JSONL is fsynced before the index
transaction commits; an unindexed complete event is replayed on the next access.
Only an incomplete final line can be discarded; malformed complete lines fail.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import re
import sqlite3
import uuid

VERSION = 'session_rollout_v1'


class RolloutSessionStore:
    def __init__(self, data_dir='data/sessions'):
        self.root = Path(data_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.rollouts = self.root / 'rollouts'
        self.rollouts.mkdir(exist_ok=True)
        self.database = self.root / 'threads.sqlite3'
        with self._connection() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS requests ('
                       'session_id TEXT NOT NULL, request_id TEXT NOT NULL, payload_sha256 TEXT NOT NULL, '
                       'status TEXT NOT NULL, result TEXT, updated_at TEXT NOT NULL, '
                       'PRIMARY KEY(session_id,request_id))')
            db.execute('CREATE TABLE IF NOT EXISTS threads ('
                       'session_id TEXT PRIMARY KEY, owner_id TEXT, updated_at TEXT, '
                       'seq INTEGER NOT NULL DEFAULT 0, offset INTEGER NOT NULL DEFAULT 0, '
                       'message_count INTEGER NOT NULL DEFAULT 0, snapshot TEXT NOT NULL DEFAULT \'{}\', '
                       'status TEXT NOT NULL DEFAULT \'active\', title TEXT NOT NULL DEFAULT \'\')')
            if 'title' not in {r['name'] for r in db.execute('PRAGMA table_info(threads)')}:
                db.execute("ALTER TABLE threads ADD COLUMN title TEXT NOT NULL DEFAULT ''")
                db.execute("UPDATE threads SET seq=0,offset=0,message_count=0,snapshot='{}'")
            if 'checkpoint_seq' not in {r['name'] for r in db.execute('PRAGMA table_info(threads)')}:
                db.execute('ALTER TABLE threads ADD COLUMN checkpoint_seq INTEGER NOT NULL DEFAULT 0')
                # This is an index migration: replay the authoritative old log
                # to recover the sequence of its most recent complete state.
                db.execute("UPDATE threads SET seq=0,offset=0,message_count=0,snapshot='{}',title=''")

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA synchronous=FULL')
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _path(self, session_id, suffix='.jsonl'):
        if not isinstance(session_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', session_id):
            raise ValueError('invalid local session ID')
        parent = self.rollouts if suffix == '.jsonl' else self.root
        path = parent / (session_id + suffix)
        if path.resolve().parent != parent.resolve():
            raise ValueError('session path escaped storage directory')
        return path

    def _sync(self, db, session_id):
        path = self._path(session_id)
        db.execute('INSERT OR IGNORE INTO threads(session_id) VALUES (?)', (session_id,))
        row = dict(db.execute('SELECT * FROM threads WHERE session_id=?', (session_id,)).fetchone())
        size = path.stat().st_size if path.exists() else 0
        if size < row['offset']:
            raise ValueError('rollout was truncated behind committed index')
        if not path.exists() or size == row['offset']:
            return row
        with path.open('r+b') as stream:
            stream.seek(row['offset'])
            while True:
                start = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith(b'\n'):
                    stream.truncate(start)
                    stream.flush()
                    os.fsync(stream.fileno())
                    break
                event = json.loads(line.decode('utf8'))
                if (event.get('version') != VERSION or event.get('session_id') != session_id
                        or event.get('seq') != row['seq'] + 1):
                    raise ValueError('invalid rollout event identity/sequence')
                kind = event['type']
                if kind == 'message':
                    row['message_count'] += 1
                    if event['role'] == 'user' and not row['title']:
                        row['title'] = event['content'][:32]
                    # Restore the recent tail even when interrupted before a checkpoint.
                    snap = json.loads(row['snapshot'])
                    tail = snap.get('history', [])
                    tail.append([event['role'], event['content']])
                    snap['history'] = tail[-12:]
                    row['snapshot'] = json.dumps(snap, ensure_ascii=False)
                elif kind == 'message_amended':
                    snap = json.loads(row['snapshot'])
                    history = snap.get('history', [])
                    if not history or history[-1] != ['assistant', event['expected']]:
                        raise ValueError('invalid assistant amendment order')
                    history[-1] = ['assistant', event['content']]
                    row['snapshot'] = json.dumps(snap, ensure_ascii=False)
                elif kind in {'checkpoint', 'legacy_import'}:
                    row['snapshot'] = json.dumps(event['snapshot'], ensure_ascii=False)
                    row['checkpoint_seq'] = event['seq']
                elif kind == 'checkpoint_patch':
                    from memory.snapshot_delta import apply, fingerprint
                    old=json.loads(row['snapshot'])
                    if fingerprint(old)!=event['base_sha256']:
                        raise ValueError('state delta base checksum mismatch')
                    updated=apply(old,event['operations'])
                    if fingerprint(updated)!=event['result_sha256']:
                        raise ValueError('state delta result checksum mismatch')
                    row['snapshot']=json.dumps(updated,ensure_ascii=False)
                elif kind == 'session_metadata':
                    owner = event['owner_id']
                    if row['owner_id'] is not None and row['owner_id'] != owner:
                        raise ValueError('session ownership changed')
                    row['owner_id'] = owner
                elif kind == 'request_started':
                    db.execute('INSERT INTO requests VALUES(?,?,?,?,?,?)',
                               (session_id, event['request_id'], event['payload_sha256'], 'pending', None, event['at']))
                elif kind == 'request_finished':
                    db.execute('UPDATE requests SET status=?,result=?,updated_at=? WHERE session_id=? AND request_id=?',
                               (event['status'], json.dumps(event.get('result'), ensure_ascii=False),
                                event['at'], session_id, event['request_id']))
                else:
                    raise ValueError('unknown rollout event type')
                row.update(seq=event['seq'], offset=stream.tell(), updated_at=event['at'])
        db.execute('UPDATE threads SET owner_id=?,updated_at=?,seq=?,offset=?,message_count=?,snapshot=?,title=?,checkpoint_seq=? WHERE session_id=?',
                   (row['owner_id'], row['updated_at'], row['seq'], row['offset'], row['message_count'], row['snapshot'], row['title'], row['checkpoint_seq'], session_id))
        return row

    def _append(self, db, session_id, payload):
        row = self._sync(db, session_id)
        event = dict(version=VERSION, session_id=session_id, seq=row['seq'] + 1,
                     event_id=str(uuid.uuid4()), at=datetime.now(timezone.utc).isoformat(), **payload)
        with self._path(session_id).open('ab') as stream:
            stream.write((json.dumps(event, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf8'))
            stream.flush()
            os.fsync(stream.fileno())
        self._sync(db, session_id)
        return event['event_id']

    def register_session(self, session_id, owner_id):
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row = self._sync(db, session_id)
            if row['owner_id'] is not None and row['owner_id'] != owner_id:
                raise ValueError('session belongs to another user')
            if row['owner_id'] is None:
                self._append(db, session_id, dict(type='session_metadata', owner_id=owner_id))
            # The old JSON file only contains the surviving tail, never claim it
            # is a full archive. Preserve the file and record that limitation.
            legacy = self._path(session_id, '.json')
            row = self._sync(db, session_id)
            events = []
            if legacy.exists():
                events = [json.loads(line) for line in self._path(session_id).read_text(encoding='utf8').splitlines()]
            imported = any(e['type'] == 'legacy_import' for e in events)
            messages = [e for e in events if e['type'] == 'message']
            import_only = all(e['type'] == 'session_metadata' or
                              (e['type'] == 'message' and e.get('source') == 'legacy_retained_tail')
                              for e in events)
            if legacy.exists() and not imported and import_only:
                snapshot = json.loads(legacy.read_text(encoding='utf8'))
                history = snapshot.get('history', [])
                prefix = [[e['role'], e['content']] for e in messages]
                if history[:len(prefix)] != prefix:
                    raise ValueError('legacy file changed during interrupted import')
                for role, content in history[len(prefix):]:
                    self._append(db, session_id, dict(type='message', role=role, content=content, source='legacy_retained_tail'))
                self._append(db, session_id, dict(type='legacy_import', snapshot=snapshot,
                                                 archive_completeness='retained_tail_only'))

    def append_message(self, session_id, role, content):
        if role not in {'user', 'assistant', 'system', 'tool'} or not isinstance(content, str):
            raise ValueError('invalid archived message')
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            return self._append(db, session_id, dict(type='message', role=role, content=content))

    @staticmethod
    def _request_id(request_id):
        if not isinstance(request_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', request_id):
            raise ValueError('invalid request ID')

    def _owned(self, db, session_id, owner_id):
        row = self._sync(db, session_id)
        if row['owner_id'] != owner_id:
            raise ValueError('session belongs to another user')

    @staticmethod
    def _request_record(row):
        if row is None:
            return None
        result = dict(row)
        result['result'] = json.loads(result['result']) if result['result'] is not None else None
        result.pop('payload_sha256', None)
        return result

    def claim_request(self, session_id, owner_id, request_id, payload):
        """At most one execution; JSONL keeps the accepted input and outcome.

        An interrupted pending claim is deliberately not retried. Exactly-once
        external effects cannot be inferred from an absent final response.
        """
        self._request_id(request_id)
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                          separators=(',', ':')).encode('utf8')).hexdigest()
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            self._owned(db, session_id, owner_id)
            row = db.execute('SELECT * FROM requests WHERE session_id=? AND request_id=?',
                             (session_id, request_id)).fetchone()
            if row is not None:
                if row['payload_sha256'] != digest:
                    raise ValueError('request ID already used with different content')
                return dict(claimed=False, **self._request_record(row))
            self._append(db, session_id, dict(type='request_started', request_id=request_id,
                                             payload_sha256=digest, payload=payload))
            return dict(claimed=True, request_id=request_id, status='pending', result=None)

    def finish_request(self, session_id, request_id, status, result=None):
        self._request_id(request_id)
        if status not in {'completed', 'failed', 'interrupted'}:
            raise ValueError('invalid terminal request status')
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            self._sync(db, session_id)
            row = db.execute('SELECT status FROM requests WHERE session_id=? AND request_id=?',
                             (session_id, request_id)).fetchone()
            if row is None:
                raise ValueError('request was not accepted')
            if row['status'] != 'pending':
                return  # A committed final result survives transport closure.
            self._append(db, session_id, dict(type='request_finished', request_id=request_id,
                                             status=status, result=result))

    def get_request(self, session_id, owner_id, request_id):
        self._request_id(request_id)
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            self._owned(db, session_id, owner_id)
            return self._request_record(db.execute('SELECT * FROM requests WHERE session_id=? AND request_id=?',
                                                   (session_id, request_id)).fetchone())

    def amend_last_assistant(self, session_id, expected, content):
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            history = json.loads(self._sync(db, session_id)['snapshot']).get('history', [])
            if not history or history[-1] != ['assistant', expected]:
                raise ValueError('assistant changed before archival amendment')
            return self._append(db, session_id, dict(type='message_amended', expected=expected, content=content))

    def save_snapshot(self, session_id, history, case_state):
        snapshot = json.loads(json.dumps(dict(history=history, case_state=case_state), ensure_ascii=False))
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row=self._sync(db, session_id)
            old=json.loads(row['snapshot'])
            if old != snapshot:
                # Normalize tuple/list serialization before comparison.
                from memory.snapshot_delta import difference, fingerprint
                full=dict(type='checkpoint',snapshot=snapshot)
                patch=dict(type='checkpoint_patch',operations=difference(old,snapshot),
                           base_sha256=fingerprint(old),result_sha256=fingerprint(snapshot))
                # Periodic full states bound recovery work; otherwise choose
                # the smaller representation, without dropping any information.
                periodic = row['seq'] - row['checkpoint_seq'] >= 32
                use_patch = bool(row['checkpoint_seq']) and not periodic and len(json.dumps(patch)) < len(json.dumps(full))
                self._append(db, session_id, patch if use_patch else full)

    def _snapshot(self, session_id):
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            return json.loads(self._sync(db, session_id)['snapshot'])

    def get_case_state(self, session_id, default_case_state):
        return deepcopy(self._snapshot(session_id).get('case_state', default_case_state))

    def get_history(self, session_id, default_history=None):
        return [tuple(x) for x in self._snapshot(session_id).get('history', default_history or [])]

    def read_events(self, session_id):
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            self._sync(db, session_id)
            path = self._path(session_id)
            return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()] if path.exists() else []

    def get_messages(self, session_id):
        messages = []
        for event in self.read_events(session_id):
            if event['type'] == 'message':
                messages.append(dict(role=event['role'], content=event['content'], event_id=event['event_id']))
            elif event['type'] == 'message_amended':
                if not messages or messages[-1]['role'] != 'assistant' or messages[-1]['content'] != event['expected']:
                    raise ValueError('invalid full archive amendment')
                messages[-1]['content'] = event['content']
        return messages

    def list_sessions(self, owner_id):
        with self._connection() as db:
            # Rebuild a missing database using the authoritative rollout files.
            db.execute('BEGIN IMMEDIATE')
            for path in self.rollouts.glob('*.jsonl'):
                self._sync(db, path.stem)
            return [dict(row) for row in db.execute(
                'SELECT session_id,title,updated_at,message_count,status FROM threads WHERE owner_id=? ORDER BY updated_at DESC', (owner_id,))]
