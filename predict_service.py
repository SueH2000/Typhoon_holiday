from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

# 初學者說明：
# 這個檔案只做一件事：把「模型檔案路徑」和「推論程式」集中管理。
# 好處是 app.py 不會塞滿模型細節，維護更簡單。

KNN_COLS = [
    'Precp','RH','StnHeight','StnPres','T.Max','T.Min','Temperature',
    'WDGust_vector_x','WDGust_vector_y','WD_vector_x','WD_vector_y','lat','lon'
]

SCALER_COLS = [
    'Dayoff','Precp','RH','StnHeight','StnPres','T.Max','T.Min','Temperature',
    'TyWS','WDGust_vector_x','WDGust_vector_y','WD_vector_x','WD_vector_y',
    'X10_radius','X7_radius','alert_num','born_spotE','born_spotN','hpa',
    'lat','lon','route_--','route_2','route_3'
]


@dataclass
class ModelBundle:
    knn: Any
    scaler: Any
    model: Any


def _resolve_path(env_key: str, default_name: str) -> Path:
    models_dir = Path(os.getenv('MODELS_DIR', 'models'))
    return Path(os.getenv(env_key, str(models_dir / default_name)))


def get_model_paths() -> Dict[str, Path]:
    return {
        'knn': _resolve_path('KNN_IMPUTER_PATH', 'kNN_imputer.joblib'),
        'scaler': _resolve_path('MINMAX_SCALER_PATH', 'MMscaler.joblib'),
        'model': _resolve_path('MODEL_PATH', 'rf_model.joblib'),
    }


def inspect_model_status() -> Dict[str, Any]:
    paths = get_model_paths()
    exists = {k: p.exists() for k, p in paths.items()}
    return {
        'paths': {k: str(v) for k, v in paths.items()},
        'exists': exists,
        'ready': all(exists.values()),
    }


def load_bundle() -> ModelBundle:
    paths = get_model_paths()
    return ModelBundle(
        knn=joblib.load(paths['knn']),
        scaler=joblib.load(paths['scaler']),
        model=joblib.load(paths['model']),
    )


def predict_probability(bundle: ModelBundle, row: Dict[str, float]) -> float:
    df = pd.DataFrame([row]).copy()

    imputed = bundle.knn.transform(df[KNN_COLS])
    df.loc[:, KNN_COLS] = imputed

    scaled = bundle.scaler.transform(df[SCALER_COLS])
    proba = float(bundle.model.predict_proba(scaled)[0][1])
    return round(proba * 100.0, 1)
