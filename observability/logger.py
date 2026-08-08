# 可观测性：统一日志配置与链路上下文
# 基于 Python 标准 logging，控制台 + 滚动文件双输出
import logging
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 日志格式：时间 | 级别 | 模块 | 消息
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# 避免 Flask debug 模式重复加载时重复添加 handler
_CONFIGURED = False
_CONFIG_LOCK = threading.Lock()

# 链路上下文：session_id 标识会话，turn_id 标识单次用户提问
# 日志和事件流统一从这里读取，保证全链路可串联
_TRACE_CONTEXT = {"session_id": "", "turn_id": ""}
_TRACE_LOCK = threading.Lock()


def setup_logging(log_dir="logs", level=logging.INFO):
    """配置根 logger：控制台 + {log_dir}/app.log（1MB x 3 滚动）。"""
    global _CONFIGURED
    with _CONFIG_LOCK:
        root = logging.getLogger()
        if _CONFIGURED:
            return root

        root.setLevel(level)
        formatter = logging.Formatter(LOG_FORMAT)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

        try:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_path / "app.log",
                maxBytes=1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError as exc:
            # 文件日志创建失败不阻断启动，仅保留控制台输出
            root.warning("File logging unavailable: %s", exc)

        _CONFIGURED = True
        return root


def get_logger(name):
    """便捷函数，获取命名 logger。"""
    return logging.getLogger(name)


def set_trace_context(session_id="", turn_id=""):
    """设置当前链路上下文，供日志和事件流统一携带。"""
    with _TRACE_LOCK:
        _TRACE_CONTEXT["session_id"] = session_id or ""
        _TRACE_CONTEXT["turn_id"] = turn_id or ""


def get_trace_context():
    """读取当前链路上下文的拷贝。"""
    with _TRACE_LOCK:
        return dict(_TRACE_CONTEXT)
