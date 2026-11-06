"""Turn a probability into a verdict. Kept free of TensorFlow so it is cheap to test."""
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "gif", "bmp"}
ALLOWED_FORMATS = {"JPEG", "MPO", "PNG", "WEBP", "GIF", "BMP"}


def build_result(p_ai, threshold, model_name, model_version, latency_ms):
    """Binary verdict uses the validation-chosen threshold; the band adds an honest 'uncertain' zone."""
    verdict = "ai" if p_ai >= threshold else "real"
    if p_ai >= threshold:
        band = "Likely AI-generated"
    elif p_ai >= 0.5:
        band = "Uncertain (leans AI)"
    else:
        band = "Likely real"
    confidence = p_ai if verdict == "ai" else 1.0 - p_ai
    return {"verdict": verdict, "band": band, "p_ai": round(p_ai, 4), "confidence": round(confidence, 4),
            "threshold": threshold, "model_name": model_name, "model_version": str(model_version),
            "latency_ms": round(latency_ms, 1)}
