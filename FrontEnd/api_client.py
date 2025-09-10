# FrontEnd/api_client.py
import os
import requests
from typing import Dict, Any, List, Optional

BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8080")
_TOKEN: Optional[str] = os.getenv("BACKEND_TOKEN")

def set_token(token: str):
    global _TOKEN
    _TOKEN = token

def _headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if _TOKEN:
        h["Authorization"] = f"Bearer {_TOKEN}"
    return h

def login(username: str, password: str) -> Dict[str, Any]:
    resp = requests.post(f"{BASE_URL}/users/login",
                         json={"username": username, "password": password},
                         headers=_headers(), timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if data.get("success") and data.get("token"):
        set_token(data["token"])
    return data

def register(username: str, password: str) -> Dict[str, Any]:
    resp = requests.post(f"{BASE_URL}/users/register",
                         json={"username": username, "password": password},
                         headers=_headers(), timeout=15)
    resp.raise_for_status()
    return resp.json()

def list_financial_data(symbol: Optional[str] = None, page: int = 0, size: int = 20) -> Dict[str, Any]:
    params = {"page": page, "size": size}
    if symbol:
        params["symbol"] = symbol
    resp = requests.get(f"{BASE_URL}/financial-data",
                        params=params, headers=_headers(), timeout=20)
    resp.raise_for_status()
    return resp.json()

def range_financial_data(symbol: str, start_time: str, end_time: str) -> List[Dict[str, Any]]:
    params = {"symbol": symbol, "startTime": start_time, "endTime": end_time}
    resp = requests.get(f"{BASE_URL}/financial-data/time-range",
                        params=params, headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()

def update_financial_data(fid: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.put(f"{BASE_URL}/financial-data/{fid}",
                        json=payload, headers=_headers(), timeout=20)
    resp.raise_for_status()
    return resp.json()

def anomalies(symbol: Optional[str] = None, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    # 你的后端目前是 GET /financial-data/anomalies（暂未实现，先跳过）
    params = {}
    if symbol: params["symbol"] = symbol
    if threshold is not None: params["threshold"] = threshold
    resp = requests.get(f"{BASE_URL}/financial-data/anomalies",
                        params=params, headers=_headers(), timeout=20)
    resp.raise_for_status()
    return resp.json()

def algorithms_available() -> List[str]:
    resp = requests.get(f"{BASE_URL}/algorithms/available",
                        headers=_headers(), timeout=15)
    resp.raise_for_status()
    return resp.json()

def predict_local(symbol: str, algorithm_name: str, prediction_type: str = "price_prediction") -> Dict[str, Any]:
    resp = requests.post(f"{BASE_URL}/algorithms/predict/local",
                         json={"symbol": symbol, "algorithmName": algorithm_name, "predictionType": prediction_type},
                         headers=_headers(), timeout=600) #一直timeout,再调大一点？
    resp.raise_for_status()
    return resp.json()

def predict_cloud(symbol: str, algorithm_name: str, prediction_type: str = "price_prediction") -> Dict[str, Any]:
    resp = requests.post(f"{BASE_URL}/algorithms/predict/cloud",
                         json={"symbol": symbol, "algorithmName": algorithm_name, "predictionType": prediction_type},
                         headers=_headers(), timeout=90)
    resp.raise_for_status()
    return resp.json()

def anomaly_detection(symbol: str, algorithm_name: str) -> Dict[str, Any]:
    resp = requests.post(f"{BASE_URL}/algorithms/anomaly-detection",
                         json={"symbol": symbol, "algorithmName": algorithm_name},
                         headers=_headers(), timeout=60)
    resp.raise_for_status()
    return resp.json()

def import_financial_data(file_path: str) -> Dict[str, Any]:
    headers = {}
    if _TOKEN:
        headers["Authorization"] = f"Bearer {_TOKEN}"
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f)}
        resp = requests.post(f"{BASE_URL}/financial-data/import",
                             files=files, headers=headers, timeout=300)
    resp.raise_for_status()
    return resp.json()

def export_financial_data(symbol: str, start_time: str, end_time: str, fmt: str = "csv") -> bytes:
    headers = {}
    if _TOKEN:
        headers["Authorization"] = f"Bearer {_TOKEN}"
    params = {"symbol": symbol, "startTime": start_time, "endTime": end_time, "format": fmt}
    resp = requests.get(f"{BASE_URL}/financial-data/export",
                        params=params, headers=headers, timeout=300)
    resp.raise_for_status()
    return resp.content

def list_symbols() -> List[str]:
    resp = requests.get(f"{BASE_URL}/financial-data/symbols",
                        headers=_headers(), timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data

