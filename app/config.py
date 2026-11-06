"""All settings come from environment variables (Task 4.4)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config():
    env = os.environ.get
    return {
        "MODEL_MODE": env("MODEL_MODE", "local"),                 # local | remote
        "MODEL_PATH": env("MODEL_PATH", str(ROOT / "models" / "champion.keras")),
        "MODEL_META_PATH": env("MODEL_META_PATH", str(ROOT / "models" / "champion_meta.json")),
        "DECISION_THRESHOLD": env("DECISION_THRESHOLD") or None,  # optional override of the meta threshold
        "DATABASE_URL": env("DATABASE_URL", f"sqlite:///{ROOT / 'visionguard_dev.db'}"),
        "MAX_UPLOAD_MB": float(env("MAX_UPLOAD_MB", "5")),
        "MODEL_API_URL": env("MODEL_API_URL", ""),
        "API_KEY": env("API_KEY", ""),
        "SECRET_KEY": env("SECRET_KEY", "dev-only-change-me"),
    }
