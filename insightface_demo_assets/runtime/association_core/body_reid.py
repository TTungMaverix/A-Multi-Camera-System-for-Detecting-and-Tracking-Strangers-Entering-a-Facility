import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from .config_loader import DEFAULT_ASSOCIATION_POLICY, deep_merge


def _merge_policy(policy):
    return deep_merge(DEFAULT_ASSOCIATION_POLICY["body_reid"], policy or {})


def _normalize(vec):
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec
    return vec / norm


def _laplacian_variance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _apply_color_normalization(image, cfg):
    normalized = image.copy()
    if bool(cfg.get("clahe_enabled", True)):
        lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        grid_size = int(cfg.get("clahe_tile_grid_size", 8))
        grid_size = max(2, grid_size)
        clahe = cv2.createCLAHE(
            clipLimit=float(cfg.get("clahe_clip_limit", 2.0)),
            tileGridSize=(grid_size, grid_size),
        )
        l_channel = clahe.apply(l_channel)
        normalized = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    if bool(cfg.get("gray_world_normalization", False)):
        channel_means = normalized.reshape(-1, 3).mean(axis=0)
        mean_gray = float(np.mean(channel_means))
        scales = np.divide(mean_gray, np.maximum(channel_means, 1e-6))
        normalized = np.clip(normalized.astype(np.float32) * scales.reshape(1, 1, 3), 0, 255).astype(np.uint8)
    return normalized


def _load_image_unicode(image_path: Path):
    try:
        data = np.fromfile(str(image_path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class OSNetBodyReIDExtractor:
    def __init__(self, policy=None):
        cfg = _merge_policy(policy)
        if not cfg.get("enabled", True):
            raise RuntimeError("Body ReID is disabled in the loaded association policy.")
        try:
            from torchreid import models
        except ImportError as exc:
            raise RuntimeError(
                "Body ReID requires torchreid to be installed. "
                "Install stranger_demo_bootstrap/requirements-demo.txt into the active runtime."
            ) from exc

        requested_device = str(cfg.get("device", "cpu")).strip().lower()
        if requested_device.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(requested_device)
            use_gpu = True
        else:
            self.device = torch.device("cpu")
            use_gpu = False

        self.cfg = cfg
        self.model = models.build_model(
            cfg.get("extractor_name", "osnet_x0_25"),
            num_classes=1000,
            pretrained=bool(cfg.get("pretrained", True)),
            use_gpu=use_gpu,
        )
        self.model.eval()
        self.model.to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def describe(self):
        return {
            "enabled": True,
            "extractor_name": self.cfg.get("extractor_name", "osnet_x0_25"),
            "device": str(self.device),
            "pretrained": bool(self.cfg.get("pretrained", True)),
            "input_width": int(self.cfg.get("input_width", 128)),
            "input_height": int(self.cfg.get("input_height", 256)),
            "tracklet_pooling_top_k": int(self.cfg.get("tracklet_pooling_top_k", 5)),
            "clahe_enabled": bool(self.cfg.get("clahe_enabled", True)),
        }

    def extract_array(self, image, source_path=""):
        result = {
            "status": "missing",
            "embedding": None,
            "message": "",
            "shape": "",
            "extractor_name": self.cfg.get("extractor_name", "osnet_x0_25"),
        }
        if image is None:
            result["message"] = f"failed_to_read:{source_path}"
            return result

        height, width = image.shape[:2]
        result["shape"] = f"{width}x{height}"
        if width < int(self.cfg.get("min_crop_width", 12)) or height < int(self.cfg.get("min_crop_height", 24)):
            result["status"] = "too_small"
            result["message"] = f"too_small:{width}x{height}"
            return result

        normalized_image = _apply_color_normalization(image, self.cfg)
        resized = cv2.resize(
            cv2.cvtColor(normalized_image, cv2.COLOR_BGR2RGB),
            (int(self.cfg.get("input_width", 128)), int(self.cfg.get("input_height", 256))),
            interpolation=cv2.INTER_LINEAR,
        )
        tensor = self.transform(resized).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            embedding = self.model(tensor)
        if isinstance(embedding, (list, tuple)):
            embedding = embedding[0]
        embedding = _normalize(embedding.detach().cpu().numpy().reshape(-1).astype(np.float32))
        result["status"] = "ok"
        result["embedding"] = embedding
        result["message"] = "ok_osnet_body_reid"
        return result

    def extract(self, image_path: Path):
        result = {
            "status": "missing",
            "embedding": None,
            "message": "",
            "shape": "",
            "extractor_name": self.cfg.get("extractor_name", "osnet_x0_25"),
        }
        if not image_path or not image_path.exists():
            result["message"] = f"missing_image:{image_path}"
            return result
        image = _load_image_unicode(image_path)
        return self.extract_array(image, source_path=str(image_path))


_EXTRACTOR_CACHE = {}


def get_body_reid_extractor(policy=None):
    cfg = _merge_policy(policy)
    cache_key = json.dumps(
        {
            "extractor_name": cfg.get("extractor_name", "osnet_x0_25"),
            "device": cfg.get("device", "cpu"),
            "pretrained": bool(cfg.get("pretrained", True)),
            "input_width": int(cfg.get("input_width", 128)),
            "input_height": int(cfg.get("input_height", 256)),
        },
        sort_keys=True,
    )
    extractor = _EXTRACTOR_CACHE.get(cache_key)
    if extractor is None:
        extractor = OSNetBodyReIDExtractor(cfg)
        _EXTRACTOR_CACHE[cache_key] = extractor
    return extractor


def _crop_quality_score(blur_score, bbox_area, reference_area):
    blur_component = min(float(blur_score), 300.0) / 300.0
    if reference_area <= 1e-6:
        area_component = 0.0
    else:
        area_component = min(float(bbox_area) / float(reference_area), 1.0)
    return round((0.65 * blur_component) + (0.35 * area_component), 4)


def build_tracklet_body_representation(crop_rows, body_reid_runtime=None, body_reid_policy=None):
    extractor = body_reid_runtime or get_body_reid_extractor(policy=body_reid_policy)
    cfg = _merge_policy(body_reid_policy or getattr(extractor, "cfg", {}))
    rows = list(crop_rows or [])
    result = {
        "status": "missing",
        "embedding": None,
        "message": "missing_tracklet_body_rows",
        "shape": "",
        "extractor_name": cfg.get("extractor_name", "osnet_x0_25"),
        "tracklet_candidate_count": len(rows),
        "tracklet_valid_crop_count": 0,
        "tracklet_selected_crop_count": 0,
        "tracklet_reject_counts": {},
        "tracklet_selected_crop_paths": [],
        "tracklet_selected_frames": [],
        "tracklet_selected_relative_secs": [],
        "tracklet_embeddings": [],
        "tracklet_pooling_strategy": "top_k_mean_pooling",
        "tracklet_quality_score": 0.0,
        "tracklet_crop_diagnostics": [],
    }
    if not rows:
        return result

    reference_bbox_area = max(float(row.get("bbox_area") or 0.0) for row in rows)
    reject_counts = Counter()
    valid_candidates = []
    diagnostics = []
    max_candidates = max(1, int(cfg.get("tracklet_pooling_max_candidates", 10)))
    min_bbox_area = float(cfg.get("tracklet_pooling_min_bbox_area", 1800.0))
    min_relative_bbox_area = float(cfg.get("tracklet_pooling_min_relative_bbox_area", 0.55))
    min_blur_score = float(cfg.get("tracklet_pooling_min_blur_score", 25.0))

    for row in rows[:max_candidates]:
        crop_path = Path(row.get("body_crop_path", ""))
        bbox_area = float(row.get("bbox_area") or 0.0)
        diagnostic = {
            "body_crop_path": str(crop_path),
            "frame_id": int(row.get("frame_id") or 0),
            "relative_sec": float(row.get("relative_sec") or 0.0),
            "bbox_area": round(bbox_area, 4),
            "accepted": False,
            "reject_reason": "",
            "blur_score": 0.0,
            "width": 0,
            "height": 0,
            "quality_score": 0.0,
        }
        if not crop_path.exists():
            diagnostic["reject_reason"] = "missing_crop"
            reject_counts["missing_crop"] += 1
            diagnostics.append(diagnostic)
            continue
        image = _load_image_unicode(crop_path)
        if image is None:
            diagnostic["reject_reason"] = "read_fail"
            reject_counts["read_fail"] += 1
            diagnostics.append(diagnostic)
            continue
        height, width = image.shape[:2]
        diagnostic["width"] = int(width)
        diagnostic["height"] = int(height)
        if width < int(cfg.get("min_crop_width", 12)) or height < int(cfg.get("min_crop_height", 24)):
            diagnostic["reject_reason"] = "too_small"
            reject_counts["too_small"] += 1
            diagnostics.append(diagnostic)
            continue
        if bbox_area and bbox_area < min_bbox_area:
            diagnostic["reject_reason"] = "bbox_too_small"
            reject_counts["bbox_too_small"] += 1
            diagnostics.append(diagnostic)
            continue
        if bbox_area and reference_bbox_area > 0.0 and (bbox_area / reference_bbox_area) < min_relative_bbox_area:
            diagnostic["reject_reason"] = "bbox_unstable"
            reject_counts["bbox_unstable"] += 1
            diagnostics.append(diagnostic)
            continue
        blur_score = _laplacian_variance(image)
        diagnostic["blur_score"] = round(blur_score, 4)
        if blur_score < min_blur_score:
            diagnostic["reject_reason"] = "blur_reject"
            reject_counts["blur_reject"] += 1
            diagnostics.append(diagnostic)
            continue
        quality_score = _crop_quality_score(blur_score, bbox_area or float(width * height), reference_bbox_area or float(width * height))
        diagnostic["accepted"] = True
        diagnostic["quality_score"] = quality_score
        diagnostics.append(diagnostic)
        valid_candidates.append(
            {
                "row": row,
                "image": image,
                "quality_score": quality_score,
                "crop_path": crop_path,
            }
        )

    result["tracklet_reject_counts"] = dict(sorted(reject_counts.items()))
    result["tracklet_crop_diagnostics"] = diagnostics
    if not valid_candidates:
        result["status"] = "no_valid_tracklet_crop"
        result["message"] = "no_valid_tracklet_crop_after_quality_filter"
        return result

    valid_candidates.sort(
        key=lambda item: (
            float(item["quality_score"]),
            float(item["row"].get("bbox_area") or 0.0),
            -int(item["row"].get("frame_id") or 0),
        ),
        reverse=True,
    )
    top_k = max(1, int(cfg.get("tracklet_pooling_top_k", 5)))
    selected_candidates = valid_candidates[:top_k]
    embeddings = []
    for candidate in selected_candidates:
        extract_result = extractor.extract_array(candidate["image"], source_path=str(candidate["crop_path"]))
        if extract_result.get("embedding") is None:
            reject_counts[extract_result.get("status") or "extract_fail"] += 1
            continue
        embeddings.append(extract_result["embedding"])
    result["tracklet_reject_counts"] = dict(sorted(reject_counts.items()))
    result["tracklet_valid_crop_count"] = len(valid_candidates)
    result["tracklet_selected_crop_count"] = len(embeddings)
    result["tracklet_selected_crop_paths"] = [str(candidate["crop_path"]) for candidate in selected_candidates]
    result["tracklet_selected_frames"] = [int(candidate["row"].get("frame_id") or 0) for candidate in selected_candidates]
    result["tracklet_selected_relative_secs"] = [float(candidate["row"].get("relative_sec") or 0.0) for candidate in selected_candidates]
    result["tracklet_embeddings"] = [np.asarray(item, dtype=np.float32) for item in embeddings]
    result["tracklet_quality_score"] = round(
        float(np.mean([candidate["quality_score"] for candidate in selected_candidates])),
        4,
    )
    if not embeddings:
        result["status"] = "tracklet_embedding_failed"
        result["message"] = "tracklet_embedding_failed"
        return result

    pooled = _normalize(np.mean(np.stack(embeddings, axis=0), axis=0))
    result["status"] = "ok"
    result["embedding"] = pooled.astype(np.float32)
    result["message"] = "ok_tracklet_body_reid"
    result["shape"] = f"pooled_{len(embeddings)}"
    return result
