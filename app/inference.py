"""Inference module (Task 4.1): load the champion model ONCE and reuse it for every request."""
import json
import time
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from app.bands import build_result
from src.data import decode_and_resize      # the SAME function used in training


class Predictor:
    def __init__(self, model_path, meta_path, threshold_override=None):
        t0 = time.time()
        self.meta = json.loads(Path(meta_path).read_text())
        self.model = keras.saving.load_model(model_path, compile=False)
        self.threshold = float(threshold_override) if threshold_override else float(self.meta["threshold"])
        self.img_size = int(self.meta.get("img_size", 224))
        self.resize = self.meta.get("resize", "pad")
        self.model_name = self.meta.get("registered_model", "visionguard")
        self.model_version = str(self.meta.get("version", "?"))
        # Warm-up call: builds the TF graph now, so the first real user is not slow
        self.model(np.zeros((1, self.img_size, self.img_size, 3), "float32"), training=False)
        self.load_seconds = time.time() - t0

    def preprocess(self, image_bytes):
        img = decode_and_resize(tf.constant(image_bytes), self.img_size, self.resize)
        return tf.cast(img, tf.float32)[None]          # shape (1, H, W, 3), pixels 0..255

    def predict(self, image_bytes, filename=None):
        t0 = time.time()
        p_ai = float(self.model(self.preprocess(image_bytes), training=False)[0, 0])
        return build_result(p_ai, self.threshold, self.model_name, self.model_version, (time.time() - t0) * 1000)
