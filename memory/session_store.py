from copy import deepcopy


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
