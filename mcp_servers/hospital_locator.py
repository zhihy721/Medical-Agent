# 地图 MCP 服务：附近医院查询（stdio transport）。
# 两种检索模式：
#   坐标模式（latitude/longitude 非零且配置 AMAP_API_KEY）：高德周边搜索
#   /v3/place/around，按直线距离排序，先查 10km，空则自动扩到 20km，取最近 5 家；
#   失败回退城市链路（有城市时）或诚实返回空。
# 城市模式数据层可切换（优先级从高到低）：
#   1. HOSPITAL_DATA_URL：显式指定的静态 JSON 端点（{"hospitals": {城市: [医院条目]}}）
#   2. AMAP_API_KEY：高德地图 Web 服务 POI 检索（真实地图数据，城市级检索）
#   3. 演示数据（默认，两者都未配置时）
# 前两者拉取失败均自动回退演示数据并在 note 中标注；返回结构统一含 source/note 字段。
import json
import os
import urllib.parse
import urllib.request

from mcp.server.mcpserver import MCPServer

# 远端数据源拉取超时：失败快速回退演示数据，不阻塞工具调用
HTTP_TIMEOUT_SECONDS = 5.0

# 高德 Web 服务：默认官方域名；AMAP_BASE_URL 仅供测试注入，非日常配置
AMAP_DEFAULT_BASE_URL = "https://restapi.amap.com"
# POI 类型 090100 = 综合医院（避免 090000 医疗服务大类带入药店/诊所）
AMAP_HOSPITAL_TYPES = "090100"

# 周边搜索半径（米）：先查 10km，空则自动扩到 20km 并在 note 标注
AMAP_NEARBY_RADIUS_METERS = 10000
AMAP_NEARBY_RADIUS_EXPANDED_METERS = 20000
# 坐标模式取距离最近的 5 家（响应已按距离升序）
AMAP_NEARBY_LIMIT = 5

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


def _parse_amap_around_pois(payload):
    """高德 place/around 响应 → 带直线距离的医院条目；响应失败（status!=1）抛异常由上层回退。

    distance 为米数字符串，换算 distance_km；为空不杜撰。其余字段同城市检索，
    POI 不含科室/急诊信息，诚实留空。
    """
    entries = []
    for poi in _parse_amap_pois(payload):
        raw_distance = str(poi.get("distance_raw") or "").strip()
        if raw_distance.isdigit():
            poi["distance_km"] = round(int(raw_distance) / 1000, 1)
        poi.pop("distance_raw", None)
        entries.append(poi)
    return entries


def _parse_amap_pois(payload):
    """高德 place/text 响应 → 医院条目列表；响应失败（status!=1）抛异常由上层回退。

    POI 不含科室/急诊/用户距离信息，诚实留空，不杜撰字段。
    周边检索的原始 distance 字段以 distance_raw 透传，由上层按需换算。
    """
    if not isinstance(payload, dict) or payload.get("status") != "1":
        raise ValueError("amap response invalid or failed")
    pois = payload.get("pois")
    if not isinstance(pois, list):
        return []
    entries = []
    for poi in pois:
        if not isinstance(poi, dict) or not poi.get("name"):
            continue
        entry = {
            "name": str(poi["name"]),
            "district": str(poi.get("adname") or ""),
            "departments": [],
            "has_emergency": False,
        }
        if "distance" in poi:
            entry["distance_raw"] = poi.get("distance")
        entries.append(entry)
    return entries


def _search_amap_around(latitude, longitude, api_key, radius_meters):
    """按经纬度 + 半径调用高德周边搜索（按距离升序），返回医院条目列表。"""
    base = os.getenv("AMAP_BASE_URL", "").strip() or AMAP_DEFAULT_BASE_URL
    query = urllib.parse.urlencode({
        "key": api_key,
        "types": AMAP_HOSPITAL_TYPES,
        # 高德坐标格式为经度在前：lng,lat
        "location": f"{longitude},{latitude}",
        "radius": str(radius_meters),
        "sortrule": "distance",
        "offset": "15",
        "page": "1",
        "extensions": "base",
    })
    with urllib.request.urlopen(f"{base}/v3/place/around?{query}", timeout=HTTP_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return _parse_amap_around_pois(payload)


def _search_amap(city, api_key):
    """按城市调用高德 POI 检索（综合医院类目），返回医院条目列表。"""
    base = os.getenv("AMAP_BASE_URL", "").strip() or AMAP_DEFAULT_BASE_URL
    query = urllib.parse.urlencode({
        "key": api_key,
        "types": AMAP_HOSPITAL_TYPES,
        "city": city,
        "citylimit": "true",
        "offset": "15",
        "page": "1",
        "extensions": "base",
    })
    with urllib.request.urlopen(f"{base}/v3/place/text?{query}", timeout=HTTP_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return _parse_amap_pois(payload)


def _load_hospitals(city=""):
    """数据层：返回 (城市→医院列表映射, 来源标记)。

    来源标记：demo（未配置外部源）/ remote（静态端点成功）/ amap（高德成功）/
    demo_fallback / amap_fallback（配置了外部源但拉取失败，回退演示数据）。
    """
    url = os.getenv("HOSPITAL_DATA_URL", "").strip()
    if url:
        try:
            with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_SECONDS) as response:
                data = json.loads(response.read().decode("utf-8"))
            hospitals = data.get("hospitals") if isinstance(data, dict) else None
            if not isinstance(hospitals, dict) or not hospitals:
                raise ValueError("hospitals mapping missing")
            return hospitals, "remote"
        except Exception:
            return _HOSPITALS, "demo_fallback"
    api_key = os.getenv("AMAP_API_KEY", "").strip()
    if api_key and city:
        try:
            return {city: _search_amap(city, api_key)}, "amap"
        except Exception:
            return _HOSPITALS, "amap_fallback"
    return _HOSPITALS, "demo"


@server.tool(
    name="search_nearby_hospitals",
    description="查询附近可前往的医院。传入 latitude/longitude 时走高德周边搜索（10km 内按距离取最近 5 家，空则自动扩到 20km）；否则按城市查询（默认演示数据；可配置 HOSPITAL_DATA_URL 静态数据源或 AMAP_API_KEY 高德地图 POI），可指定科室过滤；location 未收录时返回空列表",
)
def search_nearby_hospitals(location: str = "", department: str = "", latitude: float = 0.0, longitude: float = 0.0) -> str:
    """返回 JSON 文本：{location, count, hospitals:[{name, district, departments, has_emergency, distance_km?}], source, note, radius_km?}。"""
    city = (location or "").strip()
    api_key = os.getenv("AMAP_API_KEY", "").strip()
    has_coords = bool(latitude) and bool(longitude)
    # 坐标模式仅在无 HOSPITAL_DATA_URL 时生效：静态源无经纬度能力，配置了 URL 则维持城市链路优先
    if has_coords and api_key and not os.getenv("HOSPITAL_DATA_URL", "").strip():
        try:
            entries = _search_amap_around(latitude, longitude, api_key, AMAP_NEARBY_RADIUS_METERS)
            radius_km = AMAP_NEARBY_RADIUS_METERS // 1000
            if not entries:
                entries = _search_amap_around(latitude, longitude, api_key, AMAP_NEARBY_RADIUS_EXPANDED_METERS)
                radius_km = AMAP_NEARBY_RADIUS_EXPANDED_METERS // 1000
            note = "数据来自高德地图 POI（周边检索，按直线距离排序），就医请以实际导航与医院公告为准"
            if radius_km > AMAP_NEARBY_RADIUS_METERS // 1000:
                note += f"（{AMAP_NEARBY_RADIUS_METERS // 1000}km 内无结果，已扩大至 {radius_km}km）"
            matched = entries[:AMAP_NEARBY_LIMIT]
            return json.dumps(
                {
                    "location": city,
                    "count": len(matched),
                    "hospitals": matched,
                    "source": "amap_nearby",
                    "radius_km": radius_km,
                    "note": note,
                },
                ensure_ascii=False,
            )
        except Exception:
            if not city:
                # 坐标检索失败且无城市可回退：诚实返回空，不用无关城市的演示数据冒充就近
                return json.dumps(
                    {
                        "location": "",
                        "count": 0,
                        "hospitals": [],
                        "source": "amap_nearby_fallback",
                        "note": "高德周边检索不可用，且无城市信息可回退，请稍后重试或补充所在城市",
                    },
                    ensure_ascii=False,
                )
            # 有城市则落入下方城市链路（URL/高德城市/演示）回退

    hospitals, source = _load_hospitals(city)
    matched = []
    for key, items in hospitals.items():
        if key in city or city in key:
            matched = items
            break
    notes = {
        "demo": "演示数据，非真实地图信息，就医请以实际导航与医院公告为准",
        "remote": "数据来自配置的外部数据源，就医请以实际导航与医院公告为准",
        "amap": "数据来自高德地图 POI（城市级检索，无科室与距离明细），就医请以实际导航与医院公告为准",
        "demo_fallback": "外部数据源不可用，已回退演示数据，就医请以实际导航为准",
        "amap_fallback": "高德地图服务不可用，已回退演示数据，就医请以实际导航为准",
    }
    note = notes[source]
    if department and any(item["departments"] for item in matched):
        filtered = [item for item in matched if department in item["departments"]]
        if matched and not filtered:
            # 指定科室无匹配时返回空列表并说明，不静默回退全部（避免误导）
            matched = []
            note = f"{city or '该位置'}暂未收录带{department}的医院（演示数据）"
        else:
            matched = filtered
    # 无科室信息的数据源（高德 POI）不做科室过滤：返回空会误导用户以为无医院
    payload = {
        "location": city,
        "count": len(matched),
        "hospitals": matched,
        "source": source,
        "note": note,
    }
    return json.dumps(payload, ensure_ascii=False)


if __name__ == "__main__":
    server.run("stdio")
