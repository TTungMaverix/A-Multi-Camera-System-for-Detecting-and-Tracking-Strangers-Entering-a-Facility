import json
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
        }

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
        if image is None:
            result["message"] = f"failed_to_read:{image_path}"
            return result

        height, width = image.shape[:2]
        result["shape"] = f"{width}x{height}"
        if width < int(self.cfg.get("min_crop_width", 12)) or height < int(self.cfg.get("min_crop_height", 24)):
            result["status"] = "too_small"
            result["message"] = f"too_small:{width}x{height}"
            return result

        resized = cv2.resize(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
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
