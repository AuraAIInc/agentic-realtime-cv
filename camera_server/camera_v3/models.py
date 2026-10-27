"""YOLO model registry: downloading, lazy loading, and result -> JSON conversion.

Two instances are kept per model name:
  * "loop"    - used by the always-on detection loop with model.track(persist=True)
  * "oneshot" - used by MCP tool requests with model.predict()
Ultralytics attaches tracker callbacks to a model instance once track() has been
called, so running a one-off predict() on the loop instance would feed that frame
into the tracker and corrupt track IDs. Separate instances avoid that.

All inference happens on the detection-loop thread, so no locking is needed here.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

from camera_v3.imaging import Affine

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "default"
MODELS_CONFIG = os.path.join(os.path.dirname(__file__), "models.json")
SPEC_KEYS = ("weights", "s3_key", "cat_classes", "description")


@dataclass
class ModelSpec:
    weights: str  # local path, or an Ultralytics hub name like "yolov8s.pt" (auto-downloaded)
    s3_key: str | None = None  # if set, downloaded from the model training bucket first
    cat_classes: bool = False  # True when class indices are this user's cat IDs
    description: str = ""  # shown to MCP clients so they can pick a model
    info: dict = field(default_factory=dict)  # any other keys from the config file, passed through to clients

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "ModelSpec":
        return cls(
            weights=d.get("weights") or f"./models/{name}.pt",
            s3_key=d.get("s3_key"),
            cat_classes=bool(d.get("cat_classes", False)),
            description=d.get("description", ""),
            info={k: v for k, v in d.items() if k not in SPEC_KEYS},
        )


def load_model_config(path: str) -> dict[str, dict]:
    """name -> model settings, from the JSON models file (see models.json)."""
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict) or not all(isinstance(v, dict) for v in data.values()):
        raise ValueError(f"{path}: expected {{\"<model name>\": {{\"weights\": ..., ...}}, ...}}")
    return data


class ModelRegistry:
    def __init__(self, specs: dict[str, ModelSpec], s3_store, bucket: str, yolo_factory=None):
        self.specs = specs
        self.s3_store = s3_store
        self.bucket = bucket
        self._yolo_factory = yolo_factory
        self._instances: dict[tuple[str, str], object] = {}
        self._prepared: set[str] = set()

    @classmethod
    def from_config(cls, cfg, s3_store, yolo_factory=None) -> "ModelRegistry":
        """Models from the models file, then view-info extra_models on top (same keys).

        The default model's s3_key falls back to view-info model_s3_save_path, as in v2.
        """
        path = cfg.models_config or MODELS_CONFIG
        entries = load_model_config(path)
        for name, s in (cfg.extra_models or {}).items():
            entries[name] = {**entries.get(name, {}), **s}
        default = {"weights": "./best.pt", "cat_classes": True, **entries.pop(DEFAULT_MODEL, {})}
        default.setdefault("s3_key", cfg.model_s3_save_path)
        specs = {DEFAULT_MODEL: ModelSpec.from_dict(DEFAULT_MODEL, default)}
        specs.update((name, ModelSpec.from_dict(name, s)) for name, s in entries.items())
        logger.info(f"Models from {path}: {list(specs)}")
        return cls(specs, s3_store, cfg.model_training_bucket_name, yolo_factory)

    def names(self) -> list[str]:
        return list(self.specs)

    def spec(self, name: str) -> ModelSpec:
        if name not in self.specs:
            raise ValueError(f"unknown model {name!r}; available: {self.names()}")
        return self.specs[name]

    def describe(self, name: str) -> dict:
        """What MCP clients see about a model. classes is filled in once the model has been loaded."""
        spec = self.spec(name)
        loaded = next((m for (n, _), m in self._instances.items() if n == name), None)
        names = getattr(loaded, "names", None)
        return {
            "description": spec.description,
            "cat_classes": spec.cat_classes,
            "loaded": loaded is not None,
            "classes": {str(k): v for k, v in names.items()} if isinstance(names, dict) else None,
            **spec.info,
        }

    def _ensure_weights(self, name: str) -> str:
        spec = self.spec(name)
        if spec.s3_key and not (name != DEFAULT_MODEL and os.path.exists(spec.weights)):
            # The default model is always refreshed at startup (same as v2); extras are cached on disk.
            os.makedirs(os.path.dirname(spec.weights) or ".", exist_ok=True)
            logger.info(f"Retrieving model {name} from s3://{self.bucket}/{spec.s3_key} ...")
            self.s3_store.download(self.bucket, spec.s3_key, spec.weights)
            logger.info(f"Model {name} downloaded to {spec.weights}")
        return spec.weights

    def get(self, name: str = DEFAULT_MODEL, purpose: str = "loop"):
        key = (name, purpose)
        if key not in self._instances:
            if name not in self._prepared:
                self._ensure_weights(name)
                self._prepared.add(name)
            weights = self.spec(name).weights
            factory = self._yolo_factory
            if factory is None:
                from ultralytics import YOLO

                factory = YOLO
            self._instances[key] = factory(weights)
            logger.info(f"Loaded model {name} ({purpose}) from {weights}")
        return self._instances[key]

    def reset_tracker(self, name: str) -> None:
        model = self._instances.get((name, "loop"))
        predictor = getattr(model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            try:
                tracker.reset()
            except Exception as e:  # tracker API differs across ultralytics versions
                logger.debug(f"tracker reset failed: {e}")


def result_to_detections(result, cat_names: dict[int, str], cat_classes: bool, to_display: Affine) -> list[dict]:
    """Convert one Ultralytics result into the WebSocket detection format.

    Keeps v2's fields (cat_id, box, confidence) and adds track_id and label.
    Boxes are mapped into display (camera_img_size) coordinates.
    """
    boxes = result.boxes
    names = getattr(result, "names", {}) or {}
    cat_keys = list(cat_names.keys())
    ids = boxes.id
    out = []
    for i in range(len(boxes.cls)):
        cls = int(boxes.cls[i].item())
        if cat_classes:
            cat_id = cat_keys[cls] if cls < len(cat_keys) else -1
            label = cat_names.get(cat_id, names.get(cls, str(cls)))
        else:
            cat_id, label = -1, names.get(cls, str(cls))
        out.append(
            {
                "cat_id": cat_id,
                "box": to_display.box(boxes.xyxy[i].cpu().numpy().tolist()),
                "confidence": round(float(boxes.conf[i].item()), 4),
                "track_id": int(ids[i].item()) if ids is not None else None,
                "label": label,
            }
        )
    return out
