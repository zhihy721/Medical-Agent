# MCP 服务接入配置的加载与校验
# 沿用知识文件的校验风格：结构不合法直接抛出带定位信息的 MCPConfigError。
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MCP_CONFIG_PATH = PROJECT_ROOT / "mcp_servers.json"

# v1 仅实现 stdio；streamable_http 为预留枚举，遇到会在校验后由客户端拒绝连接
_TRANSPORTS = {"stdio", "streamable_http"}


class MCPConfigError(ValueError):
    """MCP 配置文件缺失字段或结构校验失败时抛出。"""


def load_mcp_config(path=None):
    """读取并校验 MCP 服务配置，返回归一化后的服务列表。

    归一化规则：command 为 "python" 时替换为当前解释器（跨平台/CI 友好）；
    args 中存在的相对路径 .py 文件解析为项目根目录下的绝对路径。
    配置文件不存在时返回空列表（视为未启用 MCP）。
    """
    config_path = Path(path) if path else MCP_CONFIG_PATH
    if not config_path.exists():
        return []
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MCPConfigError(f"MCP 配置读取失败 {config_path}: {exc}") from exc

    if not isinstance(data, dict) or not data.get("version"):
        raise MCPConfigError(f"MCP 配置缺少 version 字段: {config_path}")
    servers = data.get("servers", [])
    if not isinstance(servers, list):
        raise MCPConfigError("MCP 配置 servers 必须是列表")

    normalized = []
    seen_names = set()
    for index, server in enumerate(servers):
        context = f"mcp_servers.json servers[{index}]"
        if not isinstance(server, dict):
            raise MCPConfigError(f"{context} 必须是对象")
        name = server.get("name")
        if not name or not isinstance(name, str):
            raise MCPConfigError(f"{context} 缺少 name")
        if name in seen_names:
            raise MCPConfigError(f"{context} name {name} 重复")
        seen_names.add(name)

        transport = server.get("transport")
        if transport not in _TRANSPORTS:
            raise MCPConfigError(f"{context}（{name}）transport 必须是 {sorted(_TRANSPORTS)} 之一")
        if not isinstance(server.get("enabled"), bool):
            raise MCPConfigError(f"{context}（{name}）enabled 必须是布尔值")
        description = server.get("description", "")
        if not isinstance(description, str):
            raise MCPConfigError(f"{context}（{name}）description 必须是字符串")

        entry = {
            "name": name,
            "transport": transport,
            "enabled": server["enabled"],
            "description": description,
        }
        if transport == "stdio":
            command = server.get("command")
            if not command or not isinstance(command, str):
                raise MCPConfigError(f"{context}（{name}）stdio 服务缺少 command")
            args = server.get("args", [])
            if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
                raise MCPConfigError(f"{context}（{name}）args 必须是字符串列表")
            env = server.get("env", {})
            if not isinstance(env, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in env.items()
            ):
                raise MCPConfigError(f"{context}（{name}）env 必须是字符串键值字典")
            entry["command"] = sys.executable if command == "python" else command
            entry["args"] = [
                str(PROJECT_ROOT / arg) if arg.endswith(".py") and (PROJECT_ROOT / arg).exists() else arg
                for arg in args
            ]
            entry["env"] = env
        normalized.append(entry)

    return normalized
