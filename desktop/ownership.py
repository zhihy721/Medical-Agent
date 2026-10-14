"""Lifetime lease for one private L2 workspace, and conservative orphan recovery.

The OS releases the lock on actual process death. Only a newly exclusive owner
may terminate previous pending requests; it never repeats model/tool execution.
Shared web DATA_DIR is outside this recovery contract.
"""
import os
from pathlib import Path

class WorkspaceLease:
    def __init__(self,data):
        self.root=Path(data).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.stream=(self.root/'backend.lock').open('a+b');self.active=False
        if not self.stream.seek(0,2):self.stream.write(b'0');self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.active=True
        except OSError:
            self.stream.close();raise RuntimeError('此桌面工作区的后端仍在运行，请关闭原实例或使用独立工作区。') from None

    def close(self):
        if self.active:
            self.stream.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(),fcntl.LOCK_UN)
            self.active=False;self.stream.close()

    def __enter__(self):return self
    def __exit__(self,*_):self.close()

def recover_owned_pending(store,lease):
    if not isinstance(lease,WorkspaceLease) or not lease.active:raise ValueError('exclusive private workspace lease required')
    if Path(store.root).resolve()!=lease.root/'data'/'sessions':raise ValueError('recovery cannot target shared or unrelated storage')
    recovered=[]
    for path in sorted(store.rollouts.glob('*.jsonl')):
        events=store.read_events(path.stem);pending={}
        for e in events:
            if e['type']=='request_started':pending[e['request_id']]=e
            elif e['type']=='request_finished':pending.pop(e['request_id'],None)
        for id_ in pending:
            store.finish_request(path.stem,id_,'interrupted',dict(message='上次后端已退出，本次发送未完整结束。已保留归档和状态，未自动重发。',
                recovery_reason='exclusive_desktop_workspace_reacquired',recovery_pid=os.getpid(),automatic_replay=False,
                prior_tool_outcome_may_be_unknown=True))
            recovered.append(dict(session_id=path.stem,request_id=id_,status='interrupted'))
    return recovered
