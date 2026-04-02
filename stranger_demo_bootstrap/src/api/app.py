from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI

app = FastAPI(title="stranger-demo-api", version="0.1.0")


def _read_csv(path: str) -> list[dict]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    return pd.read_csv(csv_path).fillna("").to_dict(orient="records")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/events")
def events() -> list[dict]:
    return _read_csv("data/outputs/events/events.csv")


@app.get("/strangers")
def strangers() -> list[dict]:
    return _read_csv("data/db/stranger_profiles.csv")


@app.get("/runtime-summary")
def runtime_summary() -> dict:
    path = Path("data/outputs/debug/runtime_summary.json")
    if not path.exists():
        return {"status": "missing"}
    return json.loads(path.read_text(encoding="utf-8"))
