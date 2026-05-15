import numpy as np

from visimove.detection import DummyDetector, OpenCvHaarDetector
from visimove.detection.base_detector import clamp_box, safe_crop
from visimove.detection.mediapipe_detector import MediaPipeFaceMeshDetector
from visimove.pipeline.realtime_pipeline import build_detector


def test_dummy_detector_returns_face_eye_boxes_and_crops() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    result = DummyDetector().detect(frame)

    assert result.found is True
    assert result.face_box is not None
    assert result.left_eye_box is not None
    assert result.right_eye_box is not None
    assert result.left_eye_crop is not None
    assert result.right_eye_crop is not None
    assert result.eyes is not None


def test_safe_crop_clamps_to_frame_bounds() -> None:
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    crop = safe_crop(frame, (-5, -5, 12, 12))

    assert crop is not None
    assert crop.shape[:2] == (7, 7)


def test_clamp_box_rejects_empty_box() -> None:
    assert clamp_box((5, 5, 0, 10), frame_width=20, frame_height=20) is None


def test_opencv_detector_handles_no_face_safely() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    result = OpenCvHaarDetector(reuse_frames=0).detect(frame)

    assert result.found is False
    assert result.face_box is None
    assert result.left_eye_crop is None
    assert result.right_eye_crop is None


def test_mediapipe_tasks_missing_model_falls_back_to_opencv() -> None:
    detector = build_detector(
        {
            "detector_backend": "mediapipe",
            "face_landmarker_model_path": "models/detection/missing.task",
            "reuse_frames": 0,
        }
    )

    assert isinstance(detector, OpenCvHaarDetector)


def test_auto_detector_uses_safe_available_backend() -> None:
    detector = build_detector({"detector_backend": "auto", "reuse_frames": 0})

    assert isinstance(detector, OpenCvHaarDetector)


def test_mediapipe_normalized_landmarks_convert_to_pixels() -> None:
    class Point:
        def __init__(self, x: float, y: float) -> None:
            self.x = x
            self.y = y

    points = [Point(0.25, 0.5), Point(2.0, -1.0)]

    assert MediaPipeFaceMeshDetector._normalized_to_pixels(points, width=200, height=100) == [
        (50, 50),
        (200, 0),
    ]
