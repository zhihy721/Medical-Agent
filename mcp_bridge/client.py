# MCP 客户端管理器：后台单线程事件循环桥接异步 SDK 与项目的同步调用
# 连接/调用均有超时保护；单服务失败只跳过不阻断，shutdown 负责关子进程停循环。
import asyncio
import os
import sys
import threading
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from observability.logger import get_logger

_logger = get_logger("mcp_bridge.client")

CONNECT_TIMEOUT = 10.0
CALL_TIMEOUT = 15.0
# 会话失效后的重连冷却：同一服务两次重连的最小间隔，防止故障服务引发重连风暴
RECONNECT_COOLDOWN = 5.0

# 子进程环境白名单：不透传宿主全量环境（避免 API key 等敏感变量泄漏给远端服务），
# 仅保留启动 Python 子进程的平台必需键；配置里的 env 在此之上叠加（配置优先）
_ENV_PASSTHROUGH = ["PATH"]
if sys.platform == "win32":
    _ENV_PASSTHROUGH += ["SystemRoot", "SystemDrive", "TEMP", "TMP", "USERPROFILE", "PATHEXT"]
else:
    _ENV_PASSTHROUGH += ["HOME", "LANG", "LC_ALL"]


def build_server_env(extra_env=None):
    """构造 MCP 子进程的最小化环境：白名单透传 + 强制 UTF-8，再叠加配置 env。

    强制 PYTHONUTF8/PYTHONIOENCODING 保证中文内容经 stdio 传输不乱码；
    配置 env 可覆盖默认值（如服务自身需要特定编码或变量）。
    """
    env = {key: os.environ[key] for key in _ENV_PASSTHROUGH if key in os.environ}
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(extra_env or {})
    return env


class MCPClientError(RuntimeError):
    """MCP 连接或调用失败；由适配层转为 ToolResult error，不向主流程抛出。"""


class MCPClientManager:
    """管理多个 MCP 服务的连接生命周期，对外暴露同步接口。"""

    def __init__(self, connect_timeout=CONNECT_TIMEOUT, call_timeout=CALL_TIMEOUT):
        self._loop = None
        self._thread = None
        self._shutdown_event = None
        self._servers = {}
        self._configs = {}
        self._stats = {}
        self._last_reconnect_at = {}
        self._call_timeouts = {}
        self._connect_timeout = connect_timeout
        self._call_timeout = call_timeout
        self._started = False
        self._mu = threading.Lock()

    @property
    def started(self):
        return self._started

    def connected_servers(self):
        """已连接服务名列表（排序保证稳定）。"""
        return sorted(self._servers)

    # ---------- 生命周期 ----------

    def start(self, server_configs):
        """启动事件循环并逐个连接 enabled 服务，单服务失败记 warning 跳过。"""
        with self._mu:
            if self._started:
                return
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._loop.run_forever, name="mcp-client-loop", daemon=True)
            self._thread.start()
            self._shutdown_event = self._run_sync(self._make_shutdown_event(), timeout=5)
            # 服务级调用超时：未配置的服务回退全局 CALL_TIMEOUT
            self._call_timeouts = {
                config["name"]: config["call_timeout"] for config in server_configs if config.get("call_timeout")
            }
            # 保存配置副本：会话失效时据此自动重连
            self._configs = {config["name"]: config for config in server_configs}
            self._started = True

        for config in server_configs:
            try:
                self._connect_one(config)
                _logger.info("MCP 服务已连接: %s", config["name"])
            except Exception as exc:
                _logger.warning("MCP 服务 %s 连接失败，跳过: %s", config["name"], exc)

    def shutdown(self):
        """通知所有服务任务退出、等待子进程关闭并停止事件循环线程。"""
        with self._mu:
            if not self._started:
                return
            try:
                future = asyncio.run_coroutine_threadsafe(self._async_shutdown(), self._loop)
                future.result(timeout=10)
            except Exception as exc:
                _logger.warning("MCP 关闭过程异常: %s", exc)
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except RuntimeError:
                pass
            thread = self._thread
            if thread:
                thread.join(timeout=5)
            self._loop = None
            self._thread = None
            self._shutdown_event = None
            self._servers = {}
            self._configs = {}
            self._stats = {}
            self._last_reconnect_at = {}
            self._call_timeouts = {}
            self._started = False

    async def _make_shutdown_event(self):
        return asyncio.Event()

    async def _async_shutdown(self):
        self._shutdown_event.set()
        tasks = [entry["task"] for entry in self._servers.values()]
        for task in tasks:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5)
            except Exception:
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _connect_one(self, config):
        if config.get("transport") != "stdio":
            raise MCPClientError(f"v1 仅支持 stdio transport（当前 {config.get('transport')}）")
        ready = threading.Event()
        holder = {"session": None, "error": ""}
        task = self._run_sync(self._spawn_server(config, ready, holder), timeout=5)
        if not ready.wait(self._connect_timeout):
            # Task.cancel 非线程安全，必须经事件循环线程调度
            self._loop.call_soon_threadsafe(task.cancel)
            raise MCPClientError(f"连接超时（{self._connect_timeout:.0f}s）")
        if holder["session"] is None:
            self._loop.call_soon_threadsafe(task.cancel)
            raise MCPClientError(holder["error"] or "连接失败")
        self._servers[config["name"]] = {"session": holder["session"], "task": task, "config": config}
        # 指标条目：setdefault 保证重连不丢历史计数
        self._stats.setdefault(config["name"], {"calls": 0, "errors": 0, "reconnects": 0, "last_error": ""})

    async def _spawn_server(self, config, ready, holder):
        return asyncio.ensure_future(self._serve(config, ready, holder))

    async def _serve(self, config, ready, holder):
        params = StdioServerParameters(
            command=config["command"],
            args=list(config.get("args", [])),
            env=build_server_env(config.get("env")),
        )
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    holder["session"] = session
                    ready.set()
                    await self._shutdown_event.wait()
        except asyncio.CancelledError:
            ready.set()
            raise
        except Exception as exc:
            holder["error"] = str(exc)
            ready.set()

    # ---------- 同步调用接口 ----------

    def list_tools(self, server):
        """列出服务暴露的工具：[{name, description, input_schema}]；会话失效时先重连再试一次。"""
        try:
            return self._list_once(server)
        except MCPClientError:
            if self._maybe_reconnect(server):
                return self._list_once(server)
            raise

    def _list_once(self, server):
        session = self._session_for(server)
        try:
            result = self._run_sync(
                asyncio.wait_for(session.list_tools(), self._call_timeout), timeout=self._call_timeout + 5
            )
        except MCPClientError:
            raise
        except Exception as exc:
            # 会话对象已失效（如底层连接已断）时的非预期异常，归一化后由上层触发重连
            raise MCPClientError(str(exc)) from exc
        return [
            {
                "name": tool.name,
                "description": getattr(tool, "description", "") or "",
                "input_schema": getattr(tool, "inputSchema", {}) or {},
            }
            for tool in result.tools
        ]

    def call_tool(self, server, tool, arguments=None, timeout=None):
        """调用远端工具：返回解析后的数据（JSON 文本自动解析），isError 时抛 MCPClientError。

        超时优先级：显式 timeout > 服务级 call_timeout 配置 > 全局默认；
        会话失效（如子进程崩溃）时自动重连并重试一次，仍失败才抛出。
        """
        try:
            return self._call_once(server, tool, arguments, timeout)
        except MCPClientError:
            if self._maybe_reconnect(server):
                return self._call_once(server, tool, arguments, timeout)
            raise

    def _call_once(self, server, tool, arguments, timeout):
        session = self._session_for(server)
        stats = self._stats.setdefault(server, {"calls": 0, "errors": 0, "reconnects": 0, "last_error": ""})
        stats["calls"] += 1
        effective = timeout or self._call_timeouts.get(server) or self._call_timeout
        try:
            result = self._run_sync(
                asyncio.wait_for(session.call_tool(tool, arguments or {}), effective),
                timeout=effective + 5,
            )
        except MCPClientError as exc:
            stats["errors"] += 1
            stats["last_error"] = str(exc)
            raise
        except Exception as exc:
            # 会话对象已失效（如子进程崩溃后 session 不可用）时的非预期异常，
            # 归一化为 MCPClientError 由上层触发重连
            stats["errors"] += 1
            stats["last_error"] = str(exc)
            raise MCPClientError(str(exc)) from exc
        if getattr(result, "isError", False):
            stats["errors"] += 1
            stats["last_error"] = f"MCP 工具 {server}/{tool} 返回错误"
            raise MCPClientError(f"MCP 工具 {server}/{tool} 返回错误: {self._extract_text(result)}")
        return self._extract_payload(result)

    def _maybe_reconnect(self, server):
        """会话失效后按保存的配置重连：带冷却防风暴，失败只记录不抛出。"""
        entry = self._servers.get(server)
        config = (entry or {}).get("config") or self._configs.get(server)
        if not config or not self._started:
            return False
        now = time.monotonic()
        if now - self._last_reconnect_at.get(server, 0.0) < RECONNECT_COOLDOWN:
            return False
        self._last_reconnect_at[server] = now
        self._servers.pop(server, None)
        try:
            self._connect_one(config)
        except Exception as exc:
            stats = self._stats.setdefault(server, {"calls": 0, "errors": 0, "reconnects": 0, "last_error": ""})
            stats["last_error"] = f"重连失败: {exc}"
            _logger.warning("MCP 服务 %s 重连失败: %s", server, exc)
            return False
        self._stats[server]["reconnects"] += 1
        _logger.info("MCP 服务 %s 已自动重连", server)
        return True

    def status(self):
        """连接状态与调用指标：未启动/已关闭时 servers 为空，可安全调用。"""
        servers = {}
        for name in sorted(set(self._configs) | set(self._servers)):
            stats = self._stats.get(name, {})
            servers[name] = {
                "connected": name in self._servers,
                "calls": stats.get("calls", 0),
                "errors": stats.get("errors", 0),
                "reconnects": stats.get("reconnects", 0),
                "last_error": stats.get("last_error", ""),
            }
        return {"started": self._started, "servers": servers}

    def _session_for(self, server):
        if not self._started:
            raise MCPClientError("MCP 客户端未启动")
        entry = self._servers.get(server)
        if not entry:
            raise MCPClientError(f"MCP 服务 {server} 未连接")
        return entry["session"]

    def _run_sync(self, coroutine, timeout):
        if self._loop is None or not self._loop.is_running():
            raise MCPClientError("MCP 客户端未启动")
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout=timeout)
        except Exception as exc:
            future.cancel()
            raise MCPClientError(str(exc)) from exc

    @staticmethod
    def _extract_text(result):
        texts = [item.text for item in getattr(result, "content", []) if getattr(item, "text", None)]
        return "\n".join(texts)

    @staticmethod
    def _extract_payload(result):
        """工具返回内容转数据：单个 JSON 文本解析为结构，否则返回 {"text": ...}。"""
        import json

        text = MCPClientManager._extract_text(result)
        try:
            return json.loads(text)
        except (TypeError, ValueError):
            return {"text": text}


# 默认全局管理器：应用启动时 connect_mcp_servers() 接线，关闭时 shutdown
default_manager = MCPClientManager()
