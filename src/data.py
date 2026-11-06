"""
VisionGuard input pipeline (Task 2.1).

decode_and_resize() is shared by training and the Flask app, so both
apply exactly the same preprocessing. Models receive pixels in 0..255
and normalise internally, so serving cannot use the wrong scaling.
"""
from pathlib import Path

import tensorflow as tf
import keras

IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
CLASS_NAMES = {0: "real", 1: "ai"}

AUG_SETTINGS = {
    "mild":   dict(rotation=0.05, zoom=0.10, contrast=0.10),
    "strong": dict(rotation=0.15, zoom=0.25, contrast=0.25),
}


def decode_and_resize(raw_bytes, img_size=IMG_SIZE, mode="pad"):
    """Encoded image bytes -> uint8 tensor (img_size, img_size, 3).

    mode="pad":     keep aspect ratio, fill the rest with black (letterbox)  [default]
    mode="stretch": plain resize to a square (distorts aspect ratio)
    """
    img = tf.io.decode_image(raw_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    if mode == "pad":
        img = tf.image.resize_with_pad(img, img_size, img_size, antialias=True)
    elif mode == "stretch":
        img = tf.image.resize(img, (img_size, img_size), antialias=True)
    else:
        raise ValueError(f"unknown resize mode: {mode}")
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)


def load_image(path, img_size=IMG_SIZE, mode="pad"):
    return decode_and_resize(tf.io.read_file(path), img_size, mode)


def make_augmenter(strength="mild", seed=42):
    """Random augmentation for TRAINING only. Returns None for 'none'."""
    if strength == "none":
        return None
    s = AUG_SETTINGS[strength]
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=seed),
        keras.layers.RandomRotation(s["rotation"], fill_mode="constant", seed=seed),
        keras.layers.RandomZoom(s["zoom"], fill_mode="constant", seed=seed),
        keras.layers.RandomContrast(s["contrast"], seed=seed),
    ], name=f"augment_{strength}")


def read_split(split_csv, data_root):
    import pandas as pd   # training-only dependency; keeps the serving image free of pandas
    df = pd.read_csv(split_csv)
    paths = [str(Path(data_root) / p) for p in df["rel_path"]]
    return paths, df["label"].astype("int32").tolist(), df


def make_dataset(split_csv, data_root, training, img_size=IMG_SIZE,
                 batch_size=BATCH_SIZE, augment="mild", seed=42, resize="pad"):
    """Split list -> batched, prefetched tf.data.Dataset of (images 0..255 float32, labels)."""
    paths, labels, _ = read_split(split_csv, data_root)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: (load_image(p, img_size, resize), y), num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)),
                num_parallel_calls=AUTOTUNE)
    aug = make_augmenter(augment, seed) if training else None
    if aug is not None:
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)
