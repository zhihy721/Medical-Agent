from copy import deepcopy
from pathlib import Path

from memory.file_store import ensure_dir, read_json_file, write_json_file


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

    def delete_profile(self, user_id):
        # 不存在时静默返回，清除长期记忆是幂等操作
        self._profiles.pop(user_id, None)


class JsonFileProfileStore:
    """
    基于 JSON 文件的长期用户画像持久化存储。

    每个 user_id 对应一个 JSON 文件，存储在 data_dir 目录下。
    接口与 InMemoryProfileStore 完全一致，可互换使用。
    服务重启后数据不丢失。
    """

    def __init__(self, data_dir="data/profiles"):
        self._data_dir = Path(data_dir)
        ensure_dir(self._data_dir)

    def _profile_path(self, user_id):
        return self._data_dir / f"{user_id}.json"

    def get_profile(self, user_id, default_profile):
        data = read_json_file(self._profile_path(user_id))
        if data is None:
            return deepcopy(default_profile)
        return data

    def set_profile(self, user_id, profile):
        write_json_file(self._profile_path(user_id), deepcopy(profile))

    def delete_profile(self, user_id):
        # 不存在时静默返回，清除长期记忆是幂等操作
        path = self._profile_path(user_id)
        if path.exists():
            path.unlink()
