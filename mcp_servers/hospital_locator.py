# Mock 地图 MCP 服务：附近医院查询（stdio transport，纯演示数据，无外部 API）。
# 作为 MCP 接入的试点服务，验证客户端桥接、工具适配与降级链路；
# 后续接真实地图服务时，只需保持 search_nearby_hospitals 的返回结构即可无缝替换。
import json

from mcp.server.mcpserver import MCPServer

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


@server.tool(
    name="search_nearby_hospitals",
    description="按城市查询附近可前往的医院（演示数据），可指定科室过滤；location 未收录时返回空列表",
)
def search_nearby_hospitals(location: str, department: str = "") -> str:
    """返回 JSON 文本：{location, count, hospitals:[{name, district, departments, has_emergency, distance_km}]}。"""
    city = (location or "").strip()
    matched = []
    for key, items in _HOSPITALS.items():
        if key in city or city in key:
            matched = items
            break
    if department:
        filtered = [item for item in matched if department in item["departments"]]
        matched = filtered or matched
    payload = {
        "location": city,
        "count": len(matched),
        "hospitals": matched,
        "note": "演示数据，非真实地图信息，就医请以实际导航与医院公告为准",
    }
    return json.dumps(payload, ensure_ascii=False)


if __name__ == "__main__":
    server.run("stdio")
