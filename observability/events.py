# 可观测性：JSONL 结构化事件流
# 每个关键动作（LLM 调用、节点进出、工具调用、计划、评审、运行结束）落一行 JSON
# 供调试接口 /api/debug/trace 读取，也支持事后复盘与评测分析
import json
import threading
import time
from pathlib import Path

from observability.logger import get_logger, get_trace_context

_logger = get_logger("observability.events")


class EventLogger:
    """JSONL 事件写入器，未 configure 时 emit 为空操作（no-op）。"""

    def __init__(self, log_dir=None):
        self._path = None
        self._lock = threading.Lock()
        if log_dir:
            self.configure(log_dir)

    def configure(self, log_dir):
        """初始化事件文件 {log_dir}/events.jsonl，失败则保持 no-op。"""
        try:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            self._path = log_path / "events.jsonl"
        except OSError as exc:
            _logger.warning("Event stream init failed: %s", exc)
            self._path = None

    @property
    def enabled(self):
        return self._path is not None

    def emit(self, event_type, **fields):
        """追加一条事件，自动附时间戳与链路上下文。"""
        if self._path is None:
            return
        context = get_trace_context()
        event = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "session_id": context.get("session_id", ""),
            "turn_id": context.get("turn_id", ""),
            "event": event_type,
        }
        event.update(fields)
        line = json.dumps(event, ensure_ascii=False)
        with self._lock:
            try:
                with self._path.open("a", encoding="utf-8") as file:
                    file.write(line + "\n")
            except OSError as exc:
                _logger.warning("Event write failed: %s", exc)

    def read_events(self, session_id=None, limit=200):
        """读取事件，可按 session_id 过滤，返回最近的 limit 条。"""
        if self._path is None or not self._path.exists():
            return []
        events = []
        with self._path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if session_id and event.get("session_id") != session_id:
                    continue
                events.append(event)
        return events[-limit:]


# 全局单例：应用启动时调用 setup_events() 激活，
# 之前所有 emit 均为 no-op，保证测试与 CLI 场景不受影响
event_logger = EventLogger()


def setup_events(log_dir):
    """激活全局事件流，返回全局单例。"""
    event_logger.configure(log_dir)
    return event_logger
