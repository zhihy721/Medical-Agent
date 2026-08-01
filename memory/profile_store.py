from copy import deepcopy


class InMemoryProfileStore:
    """
    针对长期用户画像的最小持久化抽象

    ConversationMemory 仍然拥有合并规则和画像语义
    此存储仅按 user_id 持久化画像，使得首次提取的数据量很小
    简化了未来从数据库/Redis 迁移的过程
    """

    def __init__(self):
        self._profiles = {}

    def get_profile(self, user_id, default_profile):
        if user_id not in self._profiles:
            self._profiles[user_id] = deepcopy(default_profile)
        return deepcopy(self._profiles[user_id])

    def set_profile(self, user_id, profile):
        self._profiles[user_id] = deepcopy(profile)
