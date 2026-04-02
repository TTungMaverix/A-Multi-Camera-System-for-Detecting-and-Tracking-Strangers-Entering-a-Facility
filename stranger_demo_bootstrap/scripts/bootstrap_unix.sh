#!/usr/bin/env bash
set -euo pipefail
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONIOENCODING=utf-8

PROJECT_ROOT="${1:-.}"
VENV_NAME="${VENV_NAME:-.venv_insightface_demo}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_GPU="${USE_GPU:-0}"
PACK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

step() {
  printf '[BOOTSTRAP] %s\n' "$1"
}

ensure_dir() {
  mkdir -p "$1"
}

copy_if_missing_or_force() {
  local src="$1"
  local dst="$2"
  if [ ! -e "$dst" ] || [ "${FORCE_COPY:-0}" = "1" ]; then
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
  fi
}

mkdir -p "$PROJECT_ROOT"
ROOT="$(cd "$PROJECT_ROOT" && pwd)"

step "Project root: $ROOT"

for d in \
  "$ROOT/config" \
  "$ROOT/data/streams" \
  "$ROOT/data/known_db" \
  "$ROOT/data/models" \
  "$ROOT/data/outputs/events" \
  "$ROOT/data/outputs/snapshots" \
  "$ROOT/data/outputs/clips" \
  "$ROOT/data/outputs/logs" \
  "$ROOT/data/outputs/debug" \
  "$ROOT/data/cache" \
  "$ROOT/data/db" \
  "$ROOT/scripts" \
  "$ROOT/src/api" \
  "$ROOT/src/config" \
  "$ROOT/src/core" \
  "$ROOT/src/ui" \
  "$ROOT/tests" \
  "$ROOT/docs" \
  "$ROOT/vendor"; do
  ensure_dir "$d"
done

copy_if_missing_or_force "$PACK_ROOT/README.md" "$ROOT/README.md"
copy_if_missing_or_force "$PACK_ROOT/requirements-demo.txt" "$ROOT/requirements-demo.txt"
copy_if_missing_or_force "$PACK_ROOT/docs/CODEX_IMPLEMENTATION_BRIEF.md" "$ROOT/docs/CODEX_IMPLEMENTATION_BRIEF.md"
copy_if_missing_or_force "$PACK_ROOT/docs/CODEX_PROMPT.txt" "$ROOT/docs/CODEX_PROMPT.txt"
copy_if_missing_or_force "$PACK_ROOT/config/app.example.yaml" "$ROOT/config/app.example.yaml"
copy_if_missing_or_force "$PACK_ROOT/config/cameras.example.yaml" "$ROOT/config/cameras.example.yaml"
copy_if_missing_or_force "$PACK_ROOT/config/topology.example.yaml" "$ROOT/config/topology.example.yaml"
copy_if_missing_or_force "$PACK_ROOT/scripts/bootstrap_unix.sh" "$ROOT/scripts/bootstrap_unix.sh"
copy_if_missing_or_force "$PACK_ROOT/scripts/build_known_db.py" "$ROOT/scripts/build_known_db.py"
copy_if_missing_or_force "$PACK_ROOT/scripts/run_demo.py" "$ROOT/scripts/run_demo.py"
copy_if_missing_or_force "$PACK_ROOT/src/__init__.py" "$ROOT/src/__init__.py"
copy_if_missing_or_force "$PACK_ROOT/src/main.py" "$ROOT/src/main.py"
copy_if_missing_or_force "$PACK_ROOT/src/config/__init__.py" "$ROOT/src/config/__init__.py"
copy_if_missing_or_force "$PACK_ROOT/src/config/loader.py" "$ROOT/src/config/loader.py"
copy_if_missing_or_force "$PACK_ROOT/src/core/__init__.py" "$ROOT/src/core/__init__.py"
copy_if_missing_or_force "$PACK_ROOT/src/core/direction_filter.py" "$ROOT/src/core/direction_filter.py"
copy_if_missing_or_force "$PACK_ROOT/src/core/topology.py" "$ROOT/src/core/topology.py"
copy_if_missing_or_force "$PACK_ROOT/src/core/association.py" "$ROOT/src/core/association.py"
copy_if_missing_or_force "$PACK_ROOT/src/core/event_logger.py" "$ROOT/src/core/event_logger.py"
copy_if_missing_or_force "$PACK_ROOT/src/api/app.py" "$ROOT/src/api/app.py"
copy_if_missing_or_force "$PACK_ROOT/src/ui/dashboard.py" "$ROOT/src/ui/dashboard.py"
copy_if_missing_or_force "$PACK_ROOT/tests/test_topology.py" "$ROOT/tests/test_topology.py"
copy_if_missing_or_force "$PACK_ROOT/tests/test_association.py" "$ROOT/tests/test_association.py"

copy_if_missing_or_force "$ROOT/config/app.example.yaml" "$ROOT/config/app.yaml"
copy_if_missing_or_force "$ROOT/config/cameras.example.yaml" "$ROOT/config/cameras.yaml"
copy_if_missing_or_force "$ROOT/config/topology.example.yaml" "$ROOT/config/topology.yaml"

VENV_PATH="$ROOT/$VENV_NAME"
if [ ! -d "$VENV_PATH" ]; then
  step "Creating virtual environment"
  "$PYTHON_BIN" -m venv "$VENV_PATH"
fi

PY="$VENV_PATH/bin/python"
PIP="$VENV_PATH/bin/pip"

step "Upgrading pip tooling"
"$PY" -m ensurepip --upgrade || true
"$PY" -m pip --version >/dev/null

step "Installing demo dependencies"
"$PIP" install -r "$ROOT/requirements-demo.txt"

if [ "$USE_GPU" = "1" ]; then
  step "Switching ONNXRuntime package to GPU variant"
  "$PIP" uninstall -y onnxruntime || true
  "$PIP" install onnxruntime-gpu==1.21.0
fi

cat > "$ROOT/BOOTSTRAP_NEXT_STEPS.txt" <<'EOF'
1. Put your 4 videos or RTSP sources into config/cameras.yaml.
2. Put known identity images into data/known_db/<person_id>/.
3. Keep topology in config/topology.yaml. Set min_travel_sec = 0 for overlapping cameras.
4. Build embeddings:
   ./.venv_insightface_demo/bin/python scripts/build_known_db.py
5. Check the runtime:
   ./.venv_insightface_demo/bin/python scripts/run_demo.py
6. Optional services:
   ./.venv_insightface_demo/bin/python -m uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
   ./.venv_insightface_demo/bin/python -m streamlit run src/ui/dashboard.py --server.port 8501
EOF

step "Bootstrap complete"
printf 'Virtual environment: %s\n' "$VENV_PATH"
