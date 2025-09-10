from typing import Dict, Any, List, Optional
import os
from . import api_client

def configure(base_url: Optional[str] = None, token: Optional[str] = None):
    if base_url:
        os.environ["BACKEND_BASE_URL"] = base_url
    if token:
        os.environ["BACKEND_TOKEN"] = token
        api_client.set_token(token)

# 用户相关
def login(username: str, password: str) -> Dict[str, Any]:
    return api_client.login(username, password)

def register(username: str, password: str) -> Dict[str, Any]:
    return api_client.register(username, password)

# 行情数据相关
def list_data(symbol: Optional[str] = None, page: int = 0, size: int = 20) -> Dict[str, Any]:
    return api_client.list_financial_data(symbol, page, size)

def range_data(symbol: str, start_time: str, end_time: str) -> List[Dict[str, Any]]:
    return api_client.range_financial_data(symbol, start_time, end_time)

def update_data(fid: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    return api_client.update_financial_data(fid, payload)

def anomalies(symbol: Optional[str] = None, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    return api_client.anomalies(symbol, threshold)

def symbols() -> List[str]:
    return api_client.list_symbols()

# 算法相关
def algorithms() -> List[str]:
    return api_client.algorithms_available()

def predict_local(symbol: str, algorithm_name: str, prediction_type: str = "price_prediction") -> Dict[str, Any]:
    return api_client.predict_local(symbol, algorithm_name, prediction_type)

def predict_cloud(symbol: str, algorithm_name: str, prediction_type: str = "price_prediction") -> Dict[str, Any]:
    return api_client.predict_cloud(symbol, algorithm_name, prediction_type)

def anomaly_detection(symbol: str, algorithm_name: str) -> Dict[str, Any]:
    return api_client.anomaly_detection(symbol, algorithm_name)