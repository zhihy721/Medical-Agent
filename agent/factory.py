# 为整个系统提供一个“可切换、可扩展、可演进”的运行入口
# 导入环境变量和必要的模块
import os
# 这个模块负责根据配置创建和返回适当的Agent实例
# 支持不同的运行时（如LangGraph或经典控制器）
from agent.controller import MedicalAgent
from memory.memory import ConversationMemory

# 用环境变量控制系统运行模式，默认为LangGraph
from observability.logger import get_logger

_logger = get_logger("agent.factory")


def get_agent_runtime():
    # LangGraph是现在的默认编排运行时，保留运行时
    # 此处选择允许在调试或比较行为时显式地回退到经典控制器
    return os.getenv("AGENT_RUNTIME", "langgraph").strip().lower() or "langgraph"

# 检查LangGraph是否可用，避免在运行时才发现缺少依赖
def is_langgraph_available():
    try:
        import langgraph  # noqa: F401
    except Exception:
        return False
    return True

# 创建Agent实例的工厂函数
# 根据配置选择运行时并初始化相应的Agent
def create_agent(llm, memory=None, runtime=None, thread_id=None):
    selected_runtime = (runtime or get_agent_runtime()).lower()
    memory = memory or ConversationMemory()

    if selected_runtime == "langgraph":
        if not is_langgraph_available():
            raise ImportError(
                "AGENT_RUNTIME=langgraph but `langgraph` is not installed. "
                "Install project dependencies with `pip install -r requirements.txt`."
            )

        from agent.graph import LangGraphMedicalAgent

        agent = LangGraphMedicalAgent(llm=llm, memory=memory, thread_id=thread_id)
        agent.runtime_name = "langgraph"
        return agent

    if selected_runtime not in {"classic", "langgraph"}:
        raise ValueError(
            f"Unsupported AGENT_RUNTIME={selected_runtime!r}. "
            "Expected 'langgraph' or 'classic'."
        )

    # Classic is now an explicit compatibility fallback rather than the main path.
    _logger.warning(
        "AGENT_RUNTIME=classic is deprecated: behavior fixes (e.g. contradiction "
        "resolution) land on the LangGraph path first. Use AGENT_RUNTIME=langgraph."
    )
    agent = MedicalAgent(llm=llm, memory=memory)
    agent.runtime_name = "classic"
    return agent
