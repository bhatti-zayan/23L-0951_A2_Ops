"""App tests with a fake model, so they run in seconds and need no TensorFlow."""
import io

import pytest
from PIL import Image

from app.bands import build_result
from app.main import create_app


class FakePredictor:
    model_name, model_version, threshold = "fake", "0", 0.83

    def predict(self, data, filename=None):
        return build_result(0.9, self.threshold, self.model_name, self.model_version, 1.0)


@pytest.fixture
def client(tmp_path):
    app = create_app({"DATABASE_URL": f"sqlite:///{tmp_path / 'test.db'}", "API_KEY": "secret",
                      "MAX_UPLOAD_MB": 1}, predictor=FakePredictor())
    app.config["TESTING"] = True
    return app.test_client()


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buf, "PNG")
    return buf.getvalue()


def upload(client, data, name, url="/predict", headers=None):
    return client.post(url, data={"image": (io.BytesIO(data), name)},
                       content_type="multipart/form-data", headers=headers or {})


def test_bands():
    assert build_result(0.90, 0.83, "m", "1", 1)["band"] == "Likely AI-generated"
    assert build_result(0.70, 0.83, "m", "1", 1)["band"] == "Uncertain (leans AI)"
    assert build_result(0.20, 0.83, "m", "1", 1)["verdict"] == "real"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json["status"] == "ok" and r.json["database"] == "ok"


def test_prediction_is_logged_in_history(client):
    assert upload(client, png_bytes(), "red.png").status_code == 200
    assert b"red.png" in client.get("/history").data


def test_rejects_wrong_file_type(client):
    assert upload(client, b"just text", "notes.txt").status_code == 415


def test_rejects_undecodable_image(client):
    assert upload(client, b"this is not really a jpeg", "fake.jpg").status_code == 422


def test_rejects_too_large_file(client):
    assert upload(client, b"0" * (2 * 1024 * 1024), "big.jpg").status_code == 413


def test_sanitizes_filename(client):
    upload(client, png_bytes(), "../../etc/passwd.png")
    assert b"etc_passwd.png" in client.get("/history").data


def test_api_requires_key(client):
    assert upload(client, png_bytes(), "a.png", url="/api/predict").status_code == 401
    r = upload(client, png_bytes(), "a.png", url="/api/predict", headers={"X-API-Key": "secret"})
    assert r.status_code == 200 and r.json["verdict"] == "ai"
