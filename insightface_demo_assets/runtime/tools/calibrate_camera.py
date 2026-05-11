import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from scene_calibration import build_blank_scene_calibration, validate_scene_calibration


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def _is_int_like(value: str) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


def load_image_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise RuntimeError(f"Empty image source: {path}")
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"OpenCV could not decode image: {path}")
    return frame


def read_source_frame(source: str, frame_index=0, time_sec=None):
    source_value = str(source)
    source_path = Path(source_value)
    if source_path.exists() and source_path.suffix.lower() in IMAGE_EXTENSIONS:
        return load_image_unicode(source_path), "image"

    capture_source = int(source_value) if _is_int_like(source_value) else source_value
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open source: {source}")
    try:
        if time_sec is not None:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(time_sec)) * 1000.0)
        elif frame_index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame from source: {source}")
        return frame, "file" if not _is_int_like(source_value) else "webcam"
    finally:
        cap.release()


def normalize_point(point, width, height):
    x, y = point
    return [
        round(max(0.0, min(1.0, float(x) / float(width))), 8),
        round(max(0.0, min(1.0, float(y) / float(height))), 8),
    ]


def denormalize_point(point, width, height):
    x, y = point
    return [int(round(float(x) * width)), int(round(float(y) * height))]


def normalize_polygon(points, width, height):
    return [normalize_point(point, width, height) for point in points]


def _unit(vector):
    x, y = float(vector[0]), float(vector[1])
    norm = math.sqrt((x * x) + (y * y))
    if norm <= 1e-9:
        return [0.0, 0.0]
    return [round(x / norm, 8), round(y / norm, 8)]


def compute_in_direction_vector(p1, p2, in_side_point):
    """Return the unit normal of line p1->p2 that points toward clicked IN side.

    Runtime direction logic stores `in_side_point` and classifies points by line
    side. This vector is an explicit audit/export field: a movement vector with
    positive dot product against it moves toward the clicked facility side.
    """
    line_dx = float(p2[0]) - float(p1[0])
    line_dy = float(p2[1]) - float(p1[1])
    normal_a = [-line_dy, line_dx]
    midpoint = [(float(p1[0]) + float(p2[0])) / 2.0, (float(p1[1]) + float(p2[1])) / 2.0]
    toward_click = [float(in_side_point[0]) - midpoint[0], float(in_side_point[1]) - midpoint[1]]
    if (normal_a[0] * toward_click[0]) + (normal_a[1] * toward_click[1]) < 0:
        normal_a = [-normal_a[0], -normal_a[1]]
    return _unit(normal_a)


def classify_movement_by_direction_vector(prev_point, curr_point, in_direction_vector, epsilon=1e-6):
    movement = [float(curr_point[0]) - float(prev_point[0]), float(curr_point[1]) - float(prev_point[1])]
    score = (movement[0] * float(in_direction_vector[0])) + (movement[1] * float(in_direction_vector[1]))
    if score > epsilon:
        return "IN", round(score, 6)
    if score < -epsilon:
        return "OUT", round(score, 6)
    return "STATIONARY", round(score, 6)


def load_scene_document(path: Path | None, camera_ids=None):
    if path and path.exists():
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(text) or {}
        else:
            payload = json.loads(text)
        calibration = payload.get("scene_calibration", payload)
        if "cameras" not in calibration:
            calibration = build_blank_scene_calibration(camera_ids=camera_ids)
        return {"scene_calibration": calibration}
    return {"scene_calibration": build_blank_scene_calibration(camera_ids=camera_ids)}


def save_scene_document(path: Path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".yaml", ".yml"}:
        path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8")
    else:
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")


def build_camera_config(
    *,
    camera_id,
    source,
    source_type,
    width,
    height,
    roi_points,
    entry_points,
    anchor_point_mode="bottom_center",
    role="entry",
    description="",
    create_default_zone=True,
):
    if len(roi_points) < 3:
        raise ValueError("processing ROI requires at least 3 points")
    if len(entry_points) != 3:
        raise ValueError("entry line requires exactly 3 points: p1, p2, in-side")
    normalized_roi = normalize_polygon(roi_points, width, height)
    normalized_entry_points = normalize_polygon(entry_points, width, height)
    zone_id = f"{camera_id.lower()}_entry_main"
    cfg = {
        "camera_id": camera_id,
        "role": role,
        "description": description or f"Manual calibration for {camera_id}",
        "preview_source": str(source),
        "preview_source_type": source_type,
        "frame_size_ref": {"width": int(width), "height": int(height)},
        "anchor_point_mode": anchor_point_mode,
        "processing_roi": {"polygon": normalized_roi},
        "entry_line": {"points": normalized_entry_points[:2], "in_side_point": normalized_entry_points[2]},
        "default_zone_id": zone_id if create_default_zone else "",
        "default_subzone_id": "",
        "entry_zones": [zone_id] if create_default_zone else [],
        "exit_zones": [zone_id] if create_default_zone else [],
        "zones": [],
        "subzones": [],
    }
    if create_default_zone:
        cfg["zones"].append(
            {
                "zone_id": zone_id,
                "zone_type": "entry",
                "polygon": normalized_roi,
                "priority": 100,
                "description": "Default zone generated from the detection ROI by calibrate_camera.py.",
                "placeholder": False,
            }
        )
    return cfg


def build_per_camera_export(camera_cfg, width, height, roi_points, entry_points):
    p1, p2, in_side = entry_points
    return {
        "camera_id": camera_cfg["camera_id"],
        "resolution": [int(width), int(height)],
        "detection_roi": [[int(x), int(y)] for x, y in roi_points],
        "entry_line": {
            "p1": [int(p1[0]), int(p1[1])],
            "p2": [int(p2[0]), int(p2[1])],
            "in_side_point": [int(in_side[0]), int(in_side[1])],
            "in_direction_vector": compute_in_direction_vector(p1, p2, in_side),
        },
        "normalized": {
            "processing_roi": camera_cfg["processing_roi"],
            "entry_line": camera_cfg["entry_line"],
        },
        "runtime_compatible_camera_config": camera_cfg,
    }


class CalibrationSession:
    def __init__(self, frame, camera_id):
        self.frame = frame
        self.camera_id = camera_id
        self.mode = "roi"
        self.roi_points = []
        self.entry_points = []
        self.saved = False
        self.quit = False
        self.message = "Left click ROI points. Right click/Enter commits ROI."

    def reset_current(self):
        if self.mode == "roi":
            self.roi_points = []
            self.message = "ROI reset."
        else:
            self.entry_points = []
            self.message = "Entry line reset. Click p1, p2, then IN-side point."

    def undo(self):
        if self.mode == "roi" and self.roi_points:
            self.roi_points.pop()
        elif self.mode == "entry" and self.entry_points:
            self.entry_points.pop()
        self.message = "Undo."

    def commit_roi(self):
        if len(self.roi_points) < 3:
            self.message = "ROI needs at least 3 points."
            return
        self.mode = "entry"
        self.message = "ROI committed. Entry line: click p1, p2, then IN-side point."

    def add_point(self, x, y):
        if self.mode == "roi":
            self.roi_points.append([int(x), int(y)])
            self.message = f"ROI points: {len(self.roi_points)}"
        else:
            if len(self.entry_points) < 3:
                self.entry_points.append([int(x), int(y)])
            if len(self.entry_points) == 1:
                self.message = "Entry p1 set. Click p2."
            elif len(self.entry_points) == 2:
                self.message = "Entry line set. Click IN-side point."
            else:
                self.message = "Entry line committed. Press s to save."

    def ready_to_save(self):
        return len(self.roi_points) >= 3 and len(self.entry_points) == 3

    def draw(self):
        canvas = self.frame.copy()
        if self.roi_points:
            pts = np.asarray(self.roi_points, dtype=np.int32)
            cv2.polylines(canvas, [pts], len(self.roi_points) >= 3 and self.mode != "roi", (30, 220, 80), 2)
            for index, point in enumerate(self.roi_points):
                cv2.circle(canvas, tuple(point), 4, (30, 220, 80), -1)
                cv2.putText(canvas, f"R{index+1}", (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 220, 80), 1)
        if self.entry_points:
            for index, point in enumerate(self.entry_points):
                cv2.circle(canvas, tuple(point), 5, (0, 140, 255), -1)
                cv2.putText(canvas, ["p1", "p2", "IN"][index], (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)
            if len(self.entry_points) >= 2:
                cv2.line(canvas, tuple(self.entry_points[0]), tuple(self.entry_points[1]), (0, 140, 255), 2)
            if len(self.entry_points) == 3:
                p1, p2, in_point = self.entry_points
                mid = [int(round((p1[0] + p2[0]) / 2.0)), int(round((p1[1] + p2[1]) / 2.0))]
                vector = compute_in_direction_vector(p1, p2, in_point)
                end = [int(round(mid[0] + vector[0] * 70)), int(round(mid[1] + vector[1] * 70))]
                cv2.arrowedLine(canvas, tuple(mid), tuple(end), (255, 80, 40), 3, tipLength=0.25)
                cv2.putText(canvas, "IN", (end[0] + 6, end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 80, 40), 2)
        lines = [
            f"Camera {self.camera_id} | mode={self.mode} | left-click add | right-click/Enter commit ROI",
            "u=undo r=reset s=save q/esc=quit",
            self.message,
        ]
        for idx, text in enumerate(lines):
            y = 26 + idx * 24
            cv2.putText(canvas, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas


def _mouse_callback(event, x, y, _flags, session: CalibrationSession):
    if event == cv2.EVENT_LBUTTONDOWN:
        session.add_point(x, y)
    elif event == cv2.EVENT_RBUTTONDOWN and session.mode == "roi":
        session.commit_roi()


def run_interactive_session(frame, camera_id):
    session = CalibrationSession(frame, camera_id)
    window_name = f"Calibrate {camera_id}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, _mouse_callback, session)
    try:
        while not session.quit:
            cv2.imshow(window_name, session.draw())
            key = cv2.waitKey(20) & 0xFF
            if key in {27, ord("q")}:
                session.quit = True
            elif key in {13, 10} and session.mode == "roi":
                session.commit_roi()
            elif key == ord("u"):
                session.undo()
            elif key == ord("r"):
                session.reset_current()
            elif key == ord("s"):
                if session.ready_to_save():
                    session.saved = True
                    session.quit = True
                else:
                    session.message = "Cannot save yet: need ROI + entry line p1/p2/IN-side."
    finally:
        cv2.destroyWindow(window_name)
    if not session.saved:
        raise RuntimeError("Calibration was not saved.")
    return session.roi_points, session.entry_points


def write_outputs(args, frame, source_type, roi_points, entry_points):
    height, width = frame.shape[:2]
    output_config = Path(args.output_config)
    base_path = Path(args.base_config) if args.base_config else output_config
    document = load_scene_document(base_path if base_path.exists() else None, camera_ids=[args.camera])
    calibration = document["scene_calibration"]
    calibration.setdefault("cameras", {})
    camera_cfg = build_camera_config(
        camera_id=args.camera,
        source=args.source,
        source_type=source_type,
        width=width,
        height=height,
        roi_points=roi_points,
        entry_points=entry_points,
        anchor_point_mode=args.anchor_point_mode,
        role=args.role,
        description=args.description,
        create_default_zone=not args.no_default_zone,
    )
    calibration["cameras"][args.camera] = camera_cfg
    errors, warnings = validate_scene_calibration(calibration)
    if errors:
        raise RuntimeError("Refusing to save invalid scene calibration: " + "; ".join(errors))
    save_scene_document(output_config, {"scene_calibration": calibration})

    per_camera_json = Path(args.per_camera_json) if args.per_camera_json else output_config.with_name(f"{output_config.stem}.{args.camera}.json")
    per_camera_json.parent.mkdir(parents=True, exist_ok=True)
    per_camera_json.write_text(
        json.dumps(build_per_camera_export(camera_cfg, width, height, roi_points, entry_points), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_config, per_camera_json, warnings


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Draw processing ROI and ENTRY_IN line using OpenCV mouse input.")
    parser.add_argument("--camera", required=True, help="Camera ID to create/update, e.g. C1.")
    parser.add_argument("--source", required=True, help="Video/image/webcam index/RTSP source readable by OpenCV.")
    parser.add_argument("--output-config", required=True, help="Runtime-compatible scene calibration YAML/JSON path.")
    parser.add_argument("--base-config", default="", help="Optional existing calibration file to merge before writing output.")
    parser.add_argument("--per-camera-json", default="", help="Optional per-camera JSON export path.")
    parser.add_argument("--frame-index", type=int, default=0, help="Video frame index to calibrate on.")
    parser.add_argument("--time-sec", type=float, default=None, help="Video timestamp to calibrate on.")
    parser.add_argument("--anchor-point-mode", default="bottom_center", choices=["bottom_center", "center_center"])
    parser.add_argument("--role", default="entry")
    parser.add_argument("--description", default="")
    parser.add_argument("--no-default-zone", action="store_true", help="Do not create a default zone from the ROI polygon.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    frame, source_type = read_source_frame(args.source, frame_index=args.frame_index, time_sec=args.time_sec)
    roi_points, entry_points = run_interactive_session(frame, args.camera)
    output_config, per_camera_json, warnings = write_outputs(args, frame, source_type, roi_points, entry_points)
    print(f"SAVED_SCENE_CALIBRATION={output_config}")
    print(f"SAVED_CAMERA_JSON={per_camera_json}")
    if warnings:
        print("WARNINGS=" + "; ".join(warnings))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR={exc}", file=sys.stderr)
        raise
