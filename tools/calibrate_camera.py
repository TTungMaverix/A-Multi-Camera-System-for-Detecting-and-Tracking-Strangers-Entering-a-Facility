import sys
from pathlib import Path


RUNTIME_DIR = Path(__file__).resolve().parents[1] / "insightface_demo_assets" / "runtime"
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

from tools.calibrate_camera import main  # noqa: E402


if __name__ == "__main__":
    main()
