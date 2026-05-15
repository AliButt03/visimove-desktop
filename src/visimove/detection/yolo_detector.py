from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from visimove.detection.base_detector import Box, DetectorResult, clamp_box, safe_crop
from visimove.types import Frame


YOLO_CLASS_FACE = 0
YOLO_CLASS_LEFT_EYE = 1
YOLO_CLASS_RIGHT_EYE = 2


@dataclass
class YoloDetector:
    model_path: str
    confidence_threshold: float = 0.4
    image_size: int = 640
    device: str = "cpu"
    detect_every_n_frames: int = 3
    _model: Any | None = None
    _frame_index: int = 0
    _last_result: DetectorResult | None = None

    def __post_init__(self) -> None:
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed.") from exc
        self._model = YOLO(str(path))

    def detect(self, frame: Frame, timestamp: float | None = None) -> DetectorResult:
        self._frame_index += 1
        if (
            self._last_result is not None
            and self.detect_every_n_frames > 1
            and (self._frame_index - 1) % self.detect_every_n_frames != 0
        ):
            return self._with_reused_crops(frame, self._last_result)

        height, width = frame.shape[:2]
        results = self._model.predict(  # type: ignore[union-attr]
            frame,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )
        detections = self._parse_detections(results[0], width, height)
        result = self._build_result(frame, detections)
        self._last_result = result if result.found else None
        return result

    def _parse_detections(self, result: Any, width: int, height: int) -> dict[int, list[tuple[Box, float]]]:
        parsed: dict[int, list[tuple[Box, float]]] = {
            YOLO_CLASS_FACE: [],
            YOLO_CLASS_LEFT_EYE: [],
            YOLO_CLASS_RIGHT_EYE: [],
        }
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return parsed

        for raw_box in boxes:
            class_id = int(raw_box.cls[0].item())
            confidence = float(raw_box.conf[0].item())
            if class_id not in parsed or confidence < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = [round(value) for value in raw_box.xyxy[0].tolist()]
            box = clamp_box((x1, y1, x2 - x1, y2 - y1), width, height)
            if box is not None:
                parsed[class_id].append((box, confidence))
        return parsed

    def _build_result(self, frame: Frame, detections: dict[int, list[tuple[Box, float]]]) -> DetectorResult:
        face = self._best_box(detections[YOLO_CLASS_FACE])
        left_eye = self._best_box(detections[YOLO_CLASS_LEFT_EYE])
        right_eye = self._best_box(detections[YOLO_CLASS_RIGHT_EYE])
        confidence_values = [
            confidence
            for class_detections in detections.values()
            for _box, confidence in class_detections
        ]
        confidence = max(confidence_values, default=0.0)
        return DetectorResult(
            found=face is not None,
            face_box=face,
            left_eye_box=left_eye,
            right_eye_box=right_eye,
            left_eye_crop=safe_crop(frame, left_eye),
            right_eye_crop=safe_crop(frame, right_eye),
            landmarks={
                "source": "yolo",
                "classes": {
                    "0": "face",
                    "1": "left_eye",
                    "2": "right_eye",
                },
            },
            confidence=confidence,
        )

    @staticmethod
    def _best_box(detections: list[tuple[Box, float]]) -> Box | None:
        if not detections:
            return None
        return max(detections, key=lambda item: item[1])[0]

    @staticmethod
    def _with_reused_crops(frame: Frame, result: DetectorResult) -> DetectorResult:
        return DetectorResult(
            found=result.found,
            face_box=result.face_box,
            left_eye_box=result.left_eye_box,
            right_eye_box=result.right_eye_box,
            left_eye_crop=safe_crop(frame, result.left_eye_box),
            right_eye_crop=safe_crop(frame, result.right_eye_box),
            landmarks=result.landmarks,
            confidence=result.confidence,
            reused_previous=True,
        )

