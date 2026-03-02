from __future__ import annotations

import os
import random
from datetime import datetime, timezone
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request

from predict_service import inspect_model_status, load_bundle, predict_probability

app = Flask(__name__)

LOCATIONS: List[str] = ["臺北", "新北", "桃園", "臺中", "臺南", "高雄", "宜蘭", "花蓮"]

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


def get_weather_by_location(location_name: str) -> Dict[str, float | str]:
    seed_value = sum(ord(c) for c in location_name) + datetime.now(timezone.utc).hour
    random.seed(seed_value)
    temperature = round(random.uniform(22, 33), 1)
    rainfall = round(random.uniform(0, 180), 1)
    wind_speed = round(random.uniform(10, 65), 1)

    if rainfall > 120 or wind_speed > 50:
        description = "豪雨強風"
    elif rainfall > 60:
        description = "大雨"
    elif wind_speed > 35:
        description = "風勢偏強"
    else:
        description = "天氣尚可"

    return {
        "temperature_c": temperature,
        "rainfall_mm": rainfall,
        "wind_speed_mps": wind_speed,
        "description": description,
    }


def make_ml_feature_row(weather: Dict[str, float | str]) -> Dict[str, float]:
    # 初學者說明：
    # 這裡先把 demo 天氣映射成模型需要的欄位（24 個）。
    # 你之後接入真實氣象 + 颱風路徑資料時，再逐步替換下面常數。
    rain = float(weather["rainfall_mm"])
    temp = float(weather["temperature_c"])
    wind = float(weather["wind_speed_mps"])
    return {
        'Dayoff': 0.0,
        'Precp': rain,
        'RH': 80.0,
        'StnHeight': 30.0,
        'StnPres': 1008.0,
        'T.Max': temp + 2,
        'T.Min': temp - 2,
        'Temperature': temp,
        'TyWS': wind,
        'WDGust_vector_x': 0.10,
        'WDGust_vector_y': 0.20,
        'WD_vector_x': 0.05,
        'WD_vector_y': 0.10,
        'X10_radius': 120.0,
        'X7_radius': 240.0,
        'alert_num': 18.0,
        'born_spotE': 136.0,
        'born_spotN': 21.0,
        'hpa': 960.0,
        'lat': 25.04,
        'lon': 121.56,
        'route_--': 0.0,
        'route_2': 0.0,
        'route_3': 1.0,
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


@app.get("/app/locations")
def app_locations():
    return jsonify({"ok": True, "locations": LOCATIONS})


@app.get("/app/predict")
def app_predict():
    location_name = request.args.get("locationName", "").strip()
    if not location_name:
        return jsonify({"ok": False, "error": "請提供 locationName"}), 400
    if location_name not in LOCATIONS:
        return jsonify({"ok": False, "error": "找不到此地區，請從清單選擇。"}), 404

    weather = get_weather_by_location(location_name)
    probability = estimate_dayoff_probability(weather)
    return jsonify(
        {
            "ok": True,
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
