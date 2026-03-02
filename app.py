from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import requests
from flask import Flask, jsonify, render_template, request

from predict_service import inspect_model_status, load_bundle, predict_probability

app = Flask(__name__)

# 初學者說明：
# 這裡改成「縣市 -> 鄉鎮區 -> 經緯度」。
# 經緯度是拿來向即時天氣 API 查真實天氣資料。
COUNTY_TOWNS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "臺北市": {"中正區": (25.0324, 121.5199), "大安區": (25.0260, 121.5436), "信義區": (25.0331, 121.5662), "士林區": (25.0930, 121.5240)},
    "新北市": {"板橋區": (25.0119, 121.4628), "三重區": (25.0615, 121.4877), "新店區": (24.9676, 121.5420), "淡水區": (25.1759, 121.4436)},
    "桃園市": {"桃園區": (24.9936, 121.3010), "中壢區": (24.9650, 121.2243), "平鎮區": (24.9309, 121.2145), "大溪區": (24.8806, 121.2862)},
    "臺中市": {"西屯區": (24.1810, 120.6451), "北屯區": (24.1888, 120.7254), "豐原區": (24.2520, 120.7223), "沙鹿區": (24.2380, 120.5651)},
    "臺南市": {"中西區": (22.9967, 120.2035), "永康區": (23.0263, 120.2530), "安平區": (23.0015, 120.1602), "新營區": (23.3066, 120.3167)},
    "高雄市": {"前金區": (22.6266, 120.2930), "左營區": (22.6877, 120.2927), "鳳山區": (22.6273, 120.3581), "三民區": (22.6467, 120.3171)},
    "新竹縣": {"竹北市": (24.8387, 121.0040), "竹東鎮": (24.7362, 121.0893), "湖口鄉": (24.9000, 121.0447)},
    "苗栗縣": {"苗栗市": (24.5602, 120.8214), "頭份市": (24.6882, 120.9121), "苑裡鎮": (24.4399, 120.6530)},
    "彰化縣": {"彰化市": (24.0685, 120.5575), "員林市": (23.9569, 120.5765), "鹿港鎮": (24.0568, 120.4357)},
    "雲林縣": {"斗六市": (23.7110, 120.5416), "虎尾鎮": (23.7086, 120.4313), "北港鎮": (23.5752, 120.3030)},
    "嘉義縣": {"太保市": (23.4584, 120.3320), "朴子市": (23.4645, 120.2466), "民雄鄉": (23.5511, 120.4280)},
    "屏東縣": {"屏東市": (22.6758, 120.4925), "潮州鎮": (22.5496, 120.5422), "東港鎮": (22.4652, 120.4491)},
    "宜蘭縣": {"宜蘭市": (24.7520, 121.7545), "羅東鎮": (24.6768, 121.7669), "蘇澳鎮": (24.5942, 121.8512)},
    "花蓮縣": {"花蓮市": (23.9872, 121.6015), "吉安鄉": (23.9731, 121.5646), "玉里鎮": (23.3364, 121.3131)},
    "臺東縣": {"臺東市": (22.7583, 121.1444), "關山鎮": (23.0478, 121.1755), "成功鎮": (23.1006, 121.3650)},
}


ALL_LOCATIONS = [town for towns in COUNTY_TOWNS.values() for town in towns.keys()]

# 初學者說明：
# MODEL_MODE=ml -> 用你訓練好的模型
# MODEL_MODE=rule -> 用規則估算（沒有模型檔時也能跑）
MODEL_MODE = os.getenv("MODEL_MODE", "ml").lower()
MODEL_BUNDLE: Any = None
MODEL_LOAD_ERROR: str | None = None

if MODEL_MODE == "ml":
    try:
        MODEL_BUNDLE = load_bundle()
    except Exception as exc:  # noqa: BLE001
        MODEL_LOAD_ERROR = str(exc)


def get_weather_by_location(county: str, town: str) -> Dict[str, float | str]:
    """使用 Open-Meteo 查詢即時天氣（不需要 API key）。"""
    lat, lon = COUNTY_TOWNS[county][town]
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,precipitation,wind_speed_10m,weather_code",
            "timezone": "Asia/Taipei",
        },
        timeout=10,
    )
    response.raise_for_status()
    current = response.json().get("current", {})

    weather_code = int(current.get("weather_code", -1))
    weather_map = {
        0: "晴朗",
        1: "大致晴",
        2: "局部多雲",
        3: "陰天",
        45: "有霧",
        48: "霧淞",
        51: "毛毛雨",
        53: "細雨",
        55: "較強細雨",
        61: "小雨",
        63: "中雨",
        65: "大雨",
        71: "小雪",
        80: "陣雨",
        95: "雷雨",
    }

    return {
        "temperature_c": float(current.get("temperature_2m", 0.0)),
        "rainfall_mm": float(current.get("precipitation", 0.0)),
        "wind_speed_mps": round(float(current.get("wind_speed_10m", 0.0)) / 3.6, 1),
        "description": weather_map.get(weather_code, "未知天氣"),
    }


def make_ml_feature_row(weather: Dict[str, float | str]) -> Dict[str, float]:
    # 初學者說明：
    # 這裡先把 demo 天氣映射成模型需要的欄位（24 個）。
    # 你之後接入真實氣象 + 颱風路徑資料時，再逐步替換下面常數。
    rain = float(weather["rainfall_mm"])
    temp = float(weather["temperature_c"])
    wind = float(weather["wind_speed_mps"])
    return {
        "Dayoff": 0.0,
        "Precp": rain,
        "RH": 80.0,
        "StnHeight": 30.0,
        "StnPres": 1008.0,
        "T.Max": temp + 2,
        "T.Min": temp - 2,
        "Temperature": temp,
        "TyWS": wind,
        "WDGust_vector_x": 0.10,
        "WDGust_vector_y": 0.20,
        "WD_vector_x": 0.05,
        "WD_vector_y": 0.10,
        "X10_radius": 120.0,
        "X7_radius": 240.0,
        "alert_num": 18.0,
        "born_spotE": 136.0,
        "born_spotN": 21.0,
        "hpa": 960.0,
        "lat": 25.04,
        "lon": 121.56,
        "route_--": 0.0,
        "route_2": 0.0,
        "route_3": 1.0,
    }


def estimate_dayoff_probability_rule(weather: Dict[str, float | str]) -> float:
    rainfall_score = min(float(weather["rainfall_mm"]) / 1.6, 65)
    wind_score = min(float(weather["wind_speed_mps"]) * 0.9, 30)
    temp_score = 5 if float(weather["temperature_c"]) < 24 else 0
    return round(max(0.0, min(100.0, rainfall_score + wind_score + temp_score)), 1)


def estimate_dayoff_probability(weather: Dict[str, float | str]) -> float:
    if MODEL_MODE == "ml" and MODEL_BUNDLE is not None:
        row = make_ml_feature_row(weather)
        return predict_probability(MODEL_BUNDLE, row)
    return estimate_dayoff_probability_rule(weather)


def build_advice(probability: float) -> str:
    if probability >= 75:
        return "高機率停班停課，請提前準備物資並減少外出。"
    if probability >= 45:
        return "有機會停班停課，請持續留意縣市政府公告。"
    return "目前停班停課機率偏低，但仍建議關注最新天氣資訊。"


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def healthcheck():
    return jsonify({"ok": True, "service": "typhoon-dayoff-app", "mode": MODEL_MODE})


@app.get("/app/model-status")
def app_model_status():
    status = inspect_model_status()
    return jsonify(
        {
            "ok": True,
            "mode": MODEL_MODE,
            "loaded": MODEL_BUNDLE is not None,
            "load_error": MODEL_LOAD_ERROR,
            **status,
        }
    )


@app.get("/app/counties")
def app_counties():
    return jsonify({"ok": True, "counties": list(COUNTY_TOWNS.keys())})


@app.get("/app/locations")
def app_locations():
    county = request.args.get("county", "").strip()
    if county:
        if county not in COUNTY_TOWNS:
            return jsonify({"ok": False, "error": "找不到此縣市"}), 404
        return jsonify({"ok": True, "county": county, "locations": list(COUNTY_TOWNS[county].keys())})

    return jsonify({"ok": True, "locations": ALL_LOCATIONS})


@app.get("/app/predict")
def app_predict():
    county = request.args.get("county", "").strip()
    location_name = request.args.get("locationName", "").strip()

    if county and county not in COUNTY_TOWNS:
        return jsonify({"ok": False, "error": "找不到此縣市"}), 404
    if not county:
        return jsonify({"ok": False, "error": "請提供 county"}), 400
    if not location_name:
        return jsonify({"ok": False, "error": "請提供 locationName"}), 400

    valid_locations = list(COUNTY_TOWNS[county].keys()) if county else ALL_LOCATIONS
    if location_name not in valid_locations:
        return jsonify({"ok": False, "error": "此地點不在選擇的縣市內"}), 404

    try:
        weather = get_weather_by_location(county, location_name)
    except requests.RequestException as exc:
        return jsonify({"ok": False, "error": f"即時天氣服務暫時不可用: {exc}"}), 502

    probability = estimate_dayoff_probability(weather)
    return jsonify(
        {
            "ok": True,
            "county": county,
            "locationName": location_name,
            "weather": weather,
            "dayoff_probability": probability,
            "advice": build_advice(probability),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
