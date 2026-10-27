"""Running a model on a frame, with optional ROI crop, and mapping boxes back
to display coordinates. Shared by the detection loop and one-shot tool jobs.
Must only be called from the detection-loop thread.
"""

from __future__ import annotations

import numpy as np

from camera_v3 import imaging
from camera_v3.models import ModelRegistry, result_to_detections


class Inference:
    def __init__(self, registry: ModelRegistry, cat_names: dict[int, str], display_size: tuple[int, int], default_imgsz: tuple[int, int]):
        self.registry = registry
        self.cat_names = cat_names
        self.display_size = display_size
        self.default_imgsz = default_imgsz

    def run(
        self,
        full_frame: np.ndarray,
        model: str = "default",
        roi=None,
        imgsz=None,
        conf: float | None = None,
        track: bool = False,
    ) -> list[dict]:
        """Detect on `full_frame` (capture resolution). `roi` is in display coords.

        Without an ROI the frame is downscaled to display size first (exactly what v2 did),
        so default-mode results are unchanged. With an ROI the crop is taken from the full
        capture-resolution frame, which is where zoom gets its extra detail.
        """
        if roi:
            img, to_display = imaging.crop_display_roi(full_frame, roi, self.display_size)
        else:
            img, to_display = imaging.resize_to(full_frame, self.display_size), imaging.Affine()
        kwargs = {"verbose": False, "imgsz": list(imgsz or self.default_imgsz)}
        if conf is not None:
            kwargs["conf"] = float(conf)
        if track:
            yolo = self.registry.get(model, "loop")
            result = yolo.track(img, persist=True, **kwargs)[0]
        else:
            yolo = self.registry.get(model, "oneshot")
            result = yolo.predict(img, **kwargs)[0]
        return result_to_detections(result, self.cat_names, self.registry.spec(model).cat_classes, to_display)
