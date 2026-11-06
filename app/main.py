"""VisionGuard Flask application (Tasks 4 and 5)."""
import base64
import hmac
import io
import logging

from flask import Flask, Request, jsonify, render_template, request
from PIL import Image
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

from app.bands import ALLOWED_EXTENSIONS, ALLOWED_FORMATS
from app.config import load_config
from app.db import Store
from app.remote import RemoteModelError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("visionguard")


class InMemoryRequest(Request):
    """Keep uploads in RAM. Werkzeug normally spills uploads > 500 KB into a temp file on disk."""
    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        return io.BytesIO()


class UploadError(Exception):
    def __init__(self, title, message, status=400):
        super().__init__(message)
        self.title, self.message, self.status = title, message, status


def validate_upload(file_storage):
    """Return (safe_name, bytes) or raise UploadError. Nothing is written to disk."""
    if file_storage is None or not file_storage.filename:
        raise UploadError("No file selected", "Please choose an image to upload.", 400)
    name = secure_filename(file_storage.filename) or "upload"
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadError("Wrong file type", f"'.{ext or '?'}' files are not accepted. "
                          f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}.", 415)
    data = file_storage.read()
    if not data:
        raise UploadError("Empty file", "The uploaded file is empty.", 400)
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()                                   # structure check
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            im.load()                                     # full decode
    except Exception:
        raise UploadError("Unreadable image", f"'{name}' has an image extension but could not be decoded. "
                          "It may be corrupted or not really an image.", 422)
    if fmt not in ALLOWED_FORMATS:
        raise UploadError("Unsupported image format", f"Detected format {fmt} is not supported.", 415)
    return name, data


def thumbnail_data_uri(data, size=420):
    """Small preview embedded in the page itself, so the image is never stored."""
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def create_app(overrides=None, predictor=None):
    cfg = load_config()
    cfg.update(overrides or {})
    app = Flask(__name__)
    app.request_class = InMemoryRequest
    app.config.update(SECRET_KEY=cfg["SECRET_KEY"], MAX_CONTENT_LENGTH=int(float(cfg["MAX_UPLOAD_MB"]) * 1024 * 1024))

    if predictor is None:                                  # loaded ONCE, at start-up
        if cfg["MODEL_MODE"] == "remote":
            from app.remote import RemoteModelClient
            predictor = RemoteModelClient(cfg["MODEL_API_URL"], cfg["API_KEY"])
        else:
            from app.inference import Predictor
            predictor = Predictor(cfg["MODEL_PATH"], cfg["MODEL_META_PATH"], cfg["DECISION_THRESHOLD"])
            log.info("Model %s v%s loaded in %.1fs", predictor.model_name, predictor.model_version,
                     predictor.load_seconds)
    store = Store(cfg["DATABASE_URL"])
    source = f"remote:{cfg['MODEL_API_URL']}" if cfg["MODEL_MODE"] == "remote" else "local"

    def run_inference(name, data):
        r = predictor.predict(data, filename=name)
        store.add(filename=name, verdict=r["verdict"], band=r["band"], p_ai=r["p_ai"], confidence=r["confidence"],
                  threshold=r["threshold"], model_name=r["model_name"], model_version=str(r["model_version"]),
                  source=source, latency_ms=r["latency_ms"])
        log.info("prediction file=%s verdict=%s p_ai=%.3f", name, r["verdict"], r["p_ai"])
        return r

    @app.context_processor
    def inject_globals():
        return {"model_label": f"{predictor.model_name} v{predictor.model_version}",
                "max_mb": cfg["MAX_UPLOAD_MB"], "mode": cfg["MODEL_MODE"]}

    @app.get("/")
    def index():
        return render_template("index.html", extensions=sorted(ALLOWED_EXTENSIONS))

    @app.post("/predict")
    def predict_page():
        name, data = validate_upload(request.files.get("image"))
        result = run_inference(name, data)
        return render_template("result.html", r=result, filename=name, preview=thumbnail_data_uri(data))

    @app.get("/history")
    def history():
        return render_template("history.html", rows=store.recent())

    @app.get("/health")
    def health():
        try:
            store.ping()
            db_state = "ok"
        except Exception as e:
            db_state = f"error: {type(e).__name__}"
        body = {"status": "ok" if db_state == "ok" else "degraded", "mode": cfg["MODEL_MODE"],
                "model_name": predictor.model_name, "model_version": predictor.model_version,
                "threshold": getattr(predictor, "threshold", None), "database": db_state}
        return jsonify(body), (200 if db_state == "ok" else 503)

    @app.post("/api/predict")
    def api_predict():
        if cfg["API_KEY"] and not hmac.compare_digest(request.headers.get("X-API-Key", ""), cfg["API_KEY"]):
            return jsonify(error="Unauthorized", detail="Missing or invalid X-API-Key header."), 401
        name, data = validate_upload(request.files.get("image"))
        return jsonify(run_inference(name, data) | {"filename": name})

    def error_response(status, title, message):
        if request.path.startswith("/api/"):
            return jsonify(error=title, detail=message), status
        return render_template("error.html", title=title, message=message, status=status), status

    @app.errorhandler(UploadError)
    def handle_upload_error(e):
        return error_response(e.status, e.title, e.message)

    @app.errorhandler(RemoteModelError)
    def handle_remote_error(e):
        return error_response(502, "Model service unavailable", str(e))

    @app.errorhandler(HTTPException)
    def handle_http_error(e):
        if e.code == 413:
            return error_response(413, "File too large",
                                  f"The upload limit is {float(cfg['MAX_UPLOAD_MB']):g} MB. "
                                  "Please choose a smaller image.")
        return error_response(e.code, e.name, e.description)

    @app.errorhandler(Exception)
    def handle_unexpected(e):
        log.exception("Unhandled error")
        return error_response(500, "Something went wrong", "The server could not process this request.")

    return app
