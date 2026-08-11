# MCP 客户端管理器：后台单线程事件循环桥接异步 SDK 与项目的同步调用
# 连接/调用均有超时保护；单服务失败只跳过不阻断，shutdown 负责关子进程停循环。
import asyncio
import os
import threading

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from observability.logger import get_logger

_logger = get_logger("mcp_bridge.client")

CONNECT_TIMEOUT = 10.0
CALL_TIMEOUT = 15.0


class MCPClientError(RuntimeError):
    """MCP 连接或调用失败；由适配层转为 ToolResult error，不向主流程抛出。"""


class MCPClientManager:
    """管理多个 MCP 服务的连接生命周期，对外暴露同步接口。"""

    def __init__(self, connect_timeout=CONNECT_TIMEOUT, call_timeout=CALL_TIMEOUT):
        self._loop = None
        self._thread = None
        self._shutdown_event = None
        self._servers = {}
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
            task.cancel()
            raise MCPClientError(f"连接超时（{self._connect_timeout:.0f}s）")
        if holder["session"] is None:
            task.cancel()
            raise MCPClientError(holder["error"] or "连接失败")
        self._servers[config["name"]] = {"session": holder["session"], "task": task}

    async def _spawn_server(self, config, ready, holder):
        return asyncio.ensure_future(self._serve(config, ready, holder))

    async def _serve(self, config, ready, holder):
        params = StdioServerParameters(
            command=config["command"],
            args=list(config.get("args", [])),
            env={**os.environ, **config.get("env", {})},
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
        """列出服务暴露的工具：[{name, description, input_schema}]。"""
        session = self._session_for(server)
        result = self._run_sync(
            asyncio.wait_for(session.list_tools(), self._call_timeout), timeout=self._call_timeout + 5
        )
        return [
            {
                "name": tool.name,
                "description": getattr(tool, "description", "") or "",
                "input_schema": getattr(tool, "inputSchema", {}) or {},
            }
            for tool in result.tools
        ]

    def call_tool(self, server, tool, arguments=None):
        """调用远端工具：返回解析后的数据（JSON 文本自动解析），isError 时抛 MCPClientError。"""
        session = self._session_for(server)
        result = self._run_sync(
            asyncio.wait_for(session.call_tool(tool, arguments or {}), self._call_timeout),
            timeout=self._call_timeout + 5,
        )
        if getattr(result, "isError", False):
            raise MCPClientError(f"MCP 工具 {server}/{tool} 返回错误: {self._extract_text(result)}")
        return self._extract_payload(result)

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
