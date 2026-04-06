import sys
from pathlib import Path


RUNTIME_DIR = Path(__file__).resolve().parents[1] / "insightface_demo_assets" / "runtime"
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))
