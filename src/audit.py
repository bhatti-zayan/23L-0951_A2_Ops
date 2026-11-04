"""
VisionGuard dataset audit (Task 1.2).

Walks the dataset folder and builds one row per file with:
class, file size, width, height, colour mode, format and SHA-256 hash.
Also detects unreadable files, exact duplicates, and files that
TensorFlow's decoder cannot read (needed later by the tf.data pipeline).

Usage:
    python src/audit.py --root /content/data/raw --out reports/audit
"""
import argparse
import hashlib
import os
from pathlib import Path

import pandas as pd
from PIL import Image

# It's our own dataset, so allow very large images instead of raising
# Pillow's "decompression bomb" error.
Image.MAX_IMAGE_PIXELS = None

# The label of an image is the top-level folder it sits in.
CLASS_MAP = {"AiArtData": "ai", "RealArt": "real"}

# Format we expect for each extension (used to spot mislabelled files)
EXPECTED_FORMAT = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG",
                   ".webp": "WEBP", ".gif": "GIF"}


def sha256_of(path: Path) -> str:
    """Content hash: identical files always get identical hashes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_with_pillow(path: Path) -> dict:
    """Read size/mode/format. A file counts as readable only if it fully decodes."""
    info = {"width": None, "height": None, "mode": None, "format": None,
            "n_frames": None, "readable": False, "error": ""}
    try:
        # Pass 1: verify() checks the file structure without decoding pixels.
        with Image.open(path) as im:
            im.verify()
        # Pass 2: verify() leaves the object unusable, so reopen and decode fully.
        # This catches truncated files that verify() misses.
        with Image.open(path) as im:
            info.update(width=im.width, height=im.height, mode=im.mode,
                        format=im.format, n_frames=getattr(im, "n_frames", 1))
            im.load()
        info["readable"] = True
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"[:200]
    return info


def tf_can_decode(path: Path) -> bool:
    """Can TensorFlow's decoder (used in the tf.data pipeline) read this file?"""
    import tensorflow as tf
    try:
        tf.io.decode_image(tf.io.read_file(str(path)), channels=3,
                           expand_animations=False)
        return True
    except Exception:
        return False


def audit(root: Path, check_tf: bool = True) -> pd.DataFrame:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root)
        ext = path.suffix.lower() or "<none>"
        row = {"rel_path": str(rel),
               "class": CLASS_MAP.get(rel.parts[0], "unknown"),
               "ext": ext,
               "file_size": path.stat().st_size,
               "sha256": sha256_of(path)}
        row.update(inspect_with_pillow(path))
        row["ext_matches_format"] = (row["readable"]
                                     and EXPECTED_FORMAT.get(ext) == row["format"])
        row["tf_decodable"] = tf_can_decode(path) if (check_tf and row["readable"]) else False
        rows.append(row)
    df = pd.DataFrame(rows)

    # Exact duplicates = identical SHA-256 (byte-for-byte the same file)
    df["dup_group_size"] = df.groupby("sha256")["sha256"].transform("size")
    df["is_extra_copy"] = df.duplicated("sha256", keep="first")
    # Same image in BOTH classes = contradictory labels
    df["cross_class_dup"] = df.groupby("sha256")["class"].transform("nunique") > 1
    return df


def summarize(df: pd.DataFrame):
    counts = df.groupby("class").agg(
        files=("rel_path", "size"),
        readable=("readable", "sum"),
        corrupt=("readable", lambda s: int((~s).sum())),
        extra_copies=("is_extra_copy", "sum"),
        cross_class_dups=("cross_class_dup", "sum"),
        tf_decodable=("tf_decodable", "sum"))
    ok = df[df["readable"]]
    stats = ok.groupby("class")[["width", "height", "file_size"]].agg(["min", "median", "max"])
    return counts, stats


def main():
    ap = argparse.ArgumentParser(description="VisionGuard dataset audit")
    ap.add_argument("--root", required=True, help="dataset root folder")
    ap.add_argument("--out", default="reports/audit", help="output folder")
    ap.add_argument("--no-tf", action="store_true", help="skip TensorFlow decode check")
    args = ap.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # quieter TF logs
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")    # decoding is CPU work

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = audit(root, check_tf=not args.no_tf)
    counts, stats = summarize(df)

    df.to_csv(out / "audit_table.csv", index=False)
    counts.to_csv(out / "counts_per_class.csv")
    stats.to_csv(out / "size_stats_per_class.csv")

    pd.set_option("display.width", 200)
    print(f"Audited {len(df)} files under {root}\n")
    print("== Counts per class ==\n", counts, "\n")
    print("== Width / height / file size (readable images only) ==\n", stats, "\n")
    print("Corrupt / unreadable files :", int((~df['readable']).sum()))
    print("Duplicate groups           :", int((df.groupby('sha256').size() > 1).sum()))
    print("Extra copies (removable)   :", int(df['is_extra_copy'].sum()))
    print("Cross-class duplicates     :", int(df['cross_class_dup'].sum()))
    print("Readable but not TF-decodable:", int((df['readable'] & ~df['tf_decodable']).sum()))
    print(f"\nSaved tables to {out}/")


if __name__ == "__main__":
    main()
