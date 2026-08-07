"""
JSON 文件 I/O 基础工具。

供 JsonFileProfileStore 和 JsonFileSessionStore 复用，
封装目录创建、文件读写和删除等通用操作。
"""

import json
from pathlib import Path


def ensure_dir(path):
    """确保目录存在，不存在则递归创建。"""
    Path(path).mkdir(parents=True, exist_ok=True)


def read_json_file(path, default=None):
    """
    读取 JSON 文件。

    文件不存在或解析失败时返回 default，不抛出异常。
    """
    file_path = Path(path)
    if not file_path.exists():
        return default
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def write_json_file(path, data):
    """
    将数据写入 JSON 文件。

    自动创建父目录，使用 ensure_ascii=False 保留中文，indent=2 便于阅读。
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def delete_json_file(path):
    """删除 JSON 文件，文件不存在时静默忽略。"""
    file_path = Path(path)
    if file_path.exists():
        file_path.unlink()
