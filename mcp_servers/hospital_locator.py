# 地图 MCP 服务：附近医院查询（stdio transport）。
# 数据层可切换：默认演示数据；配置 HOSPITAL_DATA_URL 后从该端点拉取同结构 JSON
# （{"hospitals": {城市: [医院条目]}}），拉取失败自动回退演示数据并在 note 中标注。
# MCP 工具签名与返回结构不变，接入真实地图服务只需替换数据层。
import json
import os
import urllib.request

from mcp.server.mcpserver import MCPServer

# 远端数据源拉取超时：失败快速回退演示数据，不阻塞工具调用
HTTP_TIMEOUT_SECONDS = 5.0

# 演示数据：按城市/区组织，distance_km 为虚构距离
_HOSPITALS = {
    "北京": [
        {"name": "北京协和医院", "district": "东城区", "departments": ["急诊", "心内科", "综合门诊"], "has_emergency": True, "distance_km": 2.1},
        {"name": "北京大学第一医院", "district": "西城区", "departments": ["急诊", "消化内科", "综合门诊"], "has_emergency": True, "distance_km": 3.4},
        {"name": "中日友好医院", "district": "朝阳区", "departments": ["急诊", "呼吸科", "中医科"], "has_emergency": True, "distance_km": 5.8},
    ],
    "上海": [
        {"name": "复旦大学附属中山医院", "district": "徐汇区", "departments": ["急诊", "心内科", "综合门诊"], "has_emergency": True, "distance_km": 1.8},
        {"name": "上海中医药大学附属龙华医院", "district": "徐汇区", "departments": ["中医科", "脾胃病科", "综合门诊"], "has_emergency": False, "distance_km": 2.6},
        {"name": "瑞金医院", "district": "黄浦区", "departments": ["急诊", "内分泌科", "综合门诊"], "has_emergency": True, "distance_km": 4.2},
    ],
    "广州": [
        {"name": "中山大学附属第一医院", "district": "越秀区", "departments": ["急诊", "综合门诊"], "has_emergency": True, "distance_km": 2.3},
        {"name": "广东省中医院", "district": "越秀区", "departments": ["中医科", "急诊", "综合门诊"], "has_emergency": True, "distance_km": 3.1},
    ],
}

server = MCPServer("hospital_locator")


def _load_hospitals():
    """数据层：返回 (城市→医院列表映射, 来源标记)。

    来源标记：demo（未配置远端）/ remote（远端拉取成功）/
    demo_fallback（配置了远端但拉取失败，回退演示数据）。
    """
    url = os.getenv("HOSPITAL_DATA_URL", "").strip()
    if not url:
        return _HOSPITALS, "demo"
    try:
        with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
        hospitals = data.get("hospitals") if isinstance(data, dict) else None
        if not isinstance(hospitals, dict) or not hospitals:
            raise ValueError("hospitals mapping missing")
        return hospitals, "remote"
    except Exception:
        return _HOSPITALS, "demo_fallback"


@server.tool(
    name="search_nearby_hospitals",
    description="按城市查询附近可前往的医院（默认演示数据，可配置 HOSPITAL_DATA_URL 外部数据源），可指定科室过滤；location 未收录时返回空列表",
)
def search_nearby_hospitals(location: str, department: str = "") -> str:
    """返回 JSON 文本：{location, count, hospitals:[{name, district, departments, has_emergency, distance_km}]}。"""
    city = (location or "").strip()
    hospitals, source = _load_hospitals()
    matched = []
    for key, items in hospitals.items():
        if key in city or city in key:
            matched = items
            break
    notes = {
        "demo": "演示数据，非真实地图信息，就医请以实际导航与医院公告为准",
        "remote": "数据来自配置的外部数据源，就医请以实际导航与医院公告为准",
        "demo_fallback": "外部数据源不可用，已回退演示数据，就医请以实际导航为准",
    }
    note = notes[source]
    if department:
        filtered = [item for item in matched if department in item["departments"]]
        if matched and not filtered:
            # 指定科室无匹配时返回空列表并说明，不静默回退全部（避免误导）
            matched = []
            note = f"{city or '该位置'}暂未收录带{department}的医院（演示数据）"
        else:
            matched = filtered
    payload = {
        "location": city,
        "count": len(matched),
        "hospitals": matched,
        "note": note,
    }
    return json.dumps(payload, ensure_ascii=False)


if __name__ == "__main__":
    server.run("stdio")
