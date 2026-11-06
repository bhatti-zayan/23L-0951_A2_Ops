"""Console check for the inference module (Task 4.1): load once, then predict several images."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.inference import Predictor  # noqa: E402

t0 = time.time()
predictor = Predictor("models/champion.keras", "models/champion_meta.json")
print(f"Model {predictor.model_name} v{predictor.model_version} loaded ONCE in {time.time() - t0:.2f}s "
      f"(threshold={predictor.threshold}, img_size={predictor.img_size}, resize={predictor.resize})")
for path in sys.argv[1:]:
    r = predictor.predict(Path(path).read_bytes())
    print(f"{Path(path).name[:60]:60s} -> {r['band']:22s} P(AI)={r['p_ai']:.3f}  ({r['latency_ms']} ms)")
