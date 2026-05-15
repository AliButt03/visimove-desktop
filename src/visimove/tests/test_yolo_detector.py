from __future__ import annotations

import numpy as np

from visimove.detection import OpenCvHaarDetector, YoloDetector
from visimove.pipeline.realtime_pipeline import build_detector


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def tolist(self):
        return self.value

    def __getitem__(self, index):
        return self


class FakeBox:
    def __init__(self, class_id: int, confidence: float, xyxy: list[float]) -> None:
        self.cls = FakeTensor(class_id)
        self.conf = FakeTensor(confidence)
        self.xyxy = FakeTensor(xyxy)


class FakeResult:
    def __init__(self, boxes) -> None:
        self.boxes = boxes


class FakeYoloModel:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, *args, **kwargs):
        self.calls += 1
        return [
            FakeResult(
                [
                    FakeBox(0, 0.9, [10, 20, 110, 160]),
                    FakeBox(1, 0.8, [25, 55, 55, 75]),
                    FakeBox(2, 0.85, [70, 55, 100, 75]),
                ]
            )
        ]


def test_missing_yolo_weights_falls_back_to_opencv_or_mediapipe() -> None:
    detector = build_detector(
        {
            "detector_backend": "yolo",
            "yolo_model_path": "models/detection/missing.onnx",
            "reuse_frames": 0,
        }
    )

    assert isinstance(detector, OpenCvHaarDetector)


def test_yolo_detector_returns_detector_result_and_crops() -> None:
    detector = object.__new__(YoloDetector)
    detector.model_path = "unused"
    detector.confidence_threshold = 0.4
    detector.image_size = 640
    detector.device = "cpu"
    detector.detect_every_n_frames = 1
    detector._model = FakeYoloModel()
    detector._frame_index = 0
    detector._last_result = None
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    result = detector.detect(frame)

    assert result.found is True
    assert result.face_box == (10, 20, 100, 140)
    assert result.left_eye_crop is not None
    assert result.right_eye_crop is not None
    assert result.landmarks is not None
    assert result.landmarks["source"] == "yolo"


def test_yolo_detector_reuses_previous_result_between_detections() -> None:
    detector = object.__new__(YoloDetector)
    detector.model_path = "unused"
    detector.confidence_threshold = 0.4
    detector.image_size = 640
    detector.device = "cpu"
    detector.detect_every_n_frames = 3
    detector._model = FakeYoloModel()
    detector._frame_index = 0
    detector._last_result = None
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    first = detector.detect(frame)
    second = detector.detect(frame)

    assert first.reused_previous is False
    assert second.reused_previous is True
    assert detector._model.calls == 1

