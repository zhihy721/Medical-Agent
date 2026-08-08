# 工具注册表：按名字注册/查找工具，提供工具清单
# 短期用于调试面板展示，长期是接 LangGraph ToolNode / function calling 的基础
from tools.guideline_tool import get_guideline_tool
from tools.knowledge_tool import search_knowledge_tool
from tools.risk_tool import risk_assessment_tool
from tools.symptom_tool import extract_symptoms_tool


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, func):
        """注册一个被 managed_tool 装饰过的工具函数。"""
        name = getattr(func, "tool_name", None) or getattr(func, "__name__", "unknown")
        self._tools[name] = {
            "name": name,
            "version": getattr(func, "tool_version", ""),
            "description": getattr(func, "tool_description", ""),
            "func": func,
        }
        return func

    def get(self, name):
        """按名字查找工具，不存在返回 None。"""
        entry = self._tools.get(name)
        return entry["func"] if entry else None

    def list_tools(self):
        """返回工具清单（名称+版本+描述），不含函数本体，可直接 JSON 序列化。"""
        return [
            {"name": entry["name"], "version": entry["version"], "description": entry["description"]}
            for entry in self._tools.values()
        ]


# 默认注册表：模块加载时注册核心工具
default_registry = ToolRegistry()
default_registry.register(risk_assessment_tool)
default_registry.register(get_guideline_tool)
default_registry.register(extract_symptoms_tool)
default_registry.register(search_knowledge_tool)
