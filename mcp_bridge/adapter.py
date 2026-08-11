# MCP 工具适配：把远端 MCP 工具包装为本地 managed_tool 协议工具并注册进 registry
# 计时、事件流、异常转 status=error 全部复用现有协议，失败降级不阻断主流程。
from mcp_bridge.client import MCPClientError, default_manager
from mcp_bridge.config import MCPConfigError, load_mcp_config
from observability.logger import get_logger
from tools.protocol import managed_tool

_logger = get_logger("mcp_bridge.adapter")

# 远端工具统一版本标记：具体行为由服务端决定，此版本仅标识适配层协议
MCP_TOOL_VERSION = "mcp-1.0"


def build_mcp_tool(manager, server_name, tool_info):
    """把单个远端工具适配为 managed_tool 装饰的同步函数，工具名为 {server}_{tool} 防冲突。"""
    local_name = f"{server_name}_{tool_info['name']}"
    remote_name = tool_info["name"]
    description = tool_info.get("description") or f"MCP 远端工具 {server_name}/{remote_name}"

    def _invoke(**kwargs):
        return manager.call_tool(server_name, remote_name, kwargs)

    return managed_tool(local_name, MCP_TOOL_VERSION, description)(_invoke)


def register_server_tools(manager, registry, server_name, tool_infos):
    """把单个服务的远端工具批量注册进 registry，返回实际注册数。

    命名冲突防护：registry 中已存在同名工具时跳过并告警，绝不覆盖（防止远端工具
    无声顶掉本地核心工具，如 risk_assessment）。
    """
    registered = 0
    for info in tool_infos:
        local_name = f"{server_name}_{info['name']}"
        if registry.get(local_name) is not None:
            _logger.warning("MCP 工具 %s 与已注册工具同名，跳过注册（不覆盖）", local_name)
            continue
        registry.register(build_mcp_tool(manager, server_name, info))
        registered += 1
    return registered


def connect_mcp_servers(registry=None, config_path=None):
    """读取配置、连接 enabled 的 MCP 服务并把远端工具注册进 registry。

    任何失败（配置非法、服务起不来、枚举工具失败）都只降级跳过，绝不阻断应用启动。
    返回管理器实例；无可用服务时返回 None。
    """
    from tools.registry import default_registry

    registry = registry or default_registry
    try:
        servers = load_mcp_config(config_path)
    except MCPConfigError as exc:
        _logger.warning("MCP 配置非法，跳过 MCP 接入: %s", exc)
        return None

    enabled = [server for server in servers if server.get("enabled")]
    if not enabled:
        return None

    default_manager.start(enabled)
    registered = 0
    for server in enabled:
        name = server["name"]
        if name not in default_manager.connected_servers():
            continue
        try:
            tools = default_manager.list_tools(name)
        except MCPClientError as exc:
            _logger.warning("MCP 服务 %s 枚举工具失败: %s", name, exc)
            continue
        registered += register_server_tools(default_manager, registry, name, tools)

    _logger.info("MCP 接入完成，注册 %d 个远端工具", registered)
    return default_manager


def shutdown_mcp_servers():
    """应用退出时关闭所有 MCP 连接；未启动时为无操作。"""
    default_manager.shutdown()
