"""
Load the VisionGuard champion from the MLflow Model Registry BY ALIAS and classify one image.

Usage:  python scripts/predict_champion.py path/to/image.jpg
Needs env vars: MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, for src/
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import mlflow
import tensorflow as tf
from mlflow import MlflowClient

from src.data import decode_and_resize

REG_NAME, ALIAS = "23L-0951_A2_Ops_visionguard", "champion"


def main(image_path):
    mv = MlflowClient().get_model_version_by_alias(REG_NAME, ALIAS)
    thr = float(mv.tags.get("threshold", 0.5))
    size = int(mv.tags.get("img_size", 224))
    mode = mv.tags.get("resize", "pad")
    print(f"Loading {REG_NAME}@{ALIAS} -> version {mv.version} "
          f"(threshold={thr:.2f}, img_size={size}, resize={mode})")
    model = mlflow.keras.load_model(f"models:/{REG_NAME}@{ALIAS}")

    img = decode_and_resize(tf.io.read_file(image_path), size, mode)   # same preprocessing as training
    p_ai = float(model.predict(tf.cast(img, tf.float32)[None], verbose=0)[0, 0])
    label = "AI-generated" if p_ai >= thr else "Real"
    confidence = p_ai if p_ai >= thr else 1 - p_ai
    print(f"Image:   {image_path}")
    print(f"Verdict: {label}  |  P(AI) = {p_ai:.3f}  |  confidence = {confidence:.1%}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/predict_champion.py IMAGE")
    main(sys.argv[1])
