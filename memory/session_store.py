from copy import deepcopy
from pathlib import Path

from memory.file_store import ensure_dir, read_json_file, write_json_file


class InMemorySessionStore:
    """
    最小持久化摘要，用于每会话的对话状态。

    这个存储将短期状态与长期用户画像存储分开
    使得在引入数据库支持的实现之前
    session 状态和用户画像之间的边界更加明确
    """

    def __init__(self):
        self._case_states = {}
        self._histories = {}

    def get_case_state(self, session_id, default_case_state):
        if session_id not in self._case_states:
            self._case_states[session_id] = deepcopy(default_case_state)
        return deepcopy(self._case_states[session_id])

    def set_case_state(self, session_id, case_state):
        self._case_states[session_id] = deepcopy(case_state)

    def get_history(self, session_id, default_history=None):
        if session_id not in self._histories:
            self._histories[session_id] = deepcopy(default_history or [])
        return deepcopy(self._histories[session_id])

    def set_history(self, session_id, history):
        self._histories[session_id] = deepcopy(history)


class JsonFileSessionStore:
    """
    基于 JSON 文件的会话状态持久化存储。

    每个 session_id 对应一个 JSON 文件，结构为：
    {"case_state": {...}, "history": [...]}

    接口与 InMemorySessionStore 完全一致，可互换使用。
    服务重启后数据不丢失。
    """

    def __init__(self, data_dir="data/sessions"):
        self._data_dir = Path(data_dir)
        ensure_dir(self._data_dir)

    def _session_path(self, session_id):
        return self._data_dir / f"{session_id}.json"

    def _read_session(self, session_id):
        return read_json_file(self._session_path(session_id), default={})

    def _write_session(self, session_id, data):
        write_json_file(self._session_path(session_id), data)

    def get_case_state(self, session_id, default_case_state):
        data = self._read_session(session_id)
        if "case_state" not in data:
            return deepcopy(default_case_state)
        return data["case_state"]

    def set_case_state(self, session_id, case_state):
        data = self._read_session(session_id)
        data["case_state"] = deepcopy(case_state)
        self._write_session(session_id, data)

    def get_history(self, session_id, default_history=None):
        data = self._read_session(session_id)
        if "history" not in data:
            return deepcopy(default_history or [])
        return data["history"]

    def set_history(self, session_id, history):
        data = self._read_session(session_id)
        data["history"] = deepcopy(history)
        self._write_session(session_id, data)
