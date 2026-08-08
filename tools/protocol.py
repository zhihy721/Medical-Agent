# 工具协议：统一 ToolResult 返回结构与管理装饰器
# 所有工具统一返回：
# {
#   "status": "ok" | "error",
#   "data": {...},          # 工具真正产出
#   "tool": "tool_name",
#   "version": "1.0",       # 工具/规则版本，便于历史追溯
#   "error": "",            # 失败原因
#   "elapsed_ms": 12.3      # 执行耗时
# }
import time
from functools import wraps

from observability.events import event_logger
from observability.logger import get_logger

_logger = get_logger("tools.protocol")

STATUS_OK = "ok"
STATUS_ERROR = "error"


def build_tool_result(tool, version, data, status=STATUS_OK, error="", elapsed_ms=0.0):
    """构造标准 ToolResult 结构。"""
    return {
        "status": status,
        "data": data,
        "tool": tool,
        "version": version,
        "error": error,
        "elapsed_ms": round(elapsed_ms, 2),
    }


def managed_tool(name, version, description=""):
    """工具管理装饰器：计时、异常捕获（失败返回 status=error 不抛出）、自动 emit tool_call 事件。"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            try:
                data = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - started) * 1000
                result = build_tool_result(name, version, data, elapsed_ms=elapsed_ms)
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000
                _logger.warning("Tool %s failed: %s", name, exc)
                result = build_tool_result(
                    name, version, {}, status=STATUS_ERROR, error=str(exc), elapsed_ms=elapsed_ms
                )

            event_logger.emit(
                "tool_call",
                tool=name,
                version=version,
                status=result["status"],
                elapsed_ms=result["elapsed_ms"],
                error=result["error"],
            )
            return result

        # 元信息挂在包装函数上，供注册表与调试面板读取
        # 注意：协议版入口必须直接返回此包装函数，
        # 若再套一层转发函数，转发函数上没有 tool_name 等元信息
        wrapper.tool_name = name
        wrapper.tool_version = version
        wrapper.tool_description = description
        return wrapper

    return decorator


def unwrap_tool_result(result, fallback=None):
    """通用解包：status=ok 返回 data；失败记 warning 并返回降级结果。"""
    if result.get("status") == STATUS_OK:
        return result.get("data")
    _logger.warning("Tool %s failed: %s", result.get("tool"), result.get("error"))
    return fallback() if callable(fallback) else fallback
