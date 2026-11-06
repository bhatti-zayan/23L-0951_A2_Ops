"""Client for MODEL_MODE=remote: forward the image to the cloud model API (Task 8)."""
import requests


class RemoteModelError(Exception):
    pass


class RemoteModelClient:
    def __init__(self, base_url, api_key, timeout=60):
        if not base_url:
            raise ValueError("MODEL_API_URL must be set when MODEL_MODE=remote")
        self.base = base_url.rstrip("/")
        self.api_key, self.timeout = api_key, timeout
        self.model_name, self.model_version, self.threshold = "remote", "?", None
        try:                                           # best effort: learn which model the API serves
            h = requests.get(self.base + "/health", timeout=10).json()
            self.model_name = h.get("model_name", "remote")
            self.model_version = h.get("model_version", "?")
            self.threshold = h.get("threshold")
        except Exception:
            pass

    def predict(self, image_bytes, filename="upload.jpg"):
        try:
            r = requests.post(self.base + "/api/predict", files={"image": (filename, image_bytes)},
                              headers={"X-API-Key": self.api_key}, timeout=self.timeout)
        except requests.RequestException as e:
            raise RemoteModelError(f"Model API unreachable: {type(e).__name__}")
        if r.status_code != 200:
            raise RemoteModelError(f"Model API returned {r.status_code}: {r.text[:200]}")
        return r.json()
