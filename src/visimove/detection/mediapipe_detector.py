from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from visimove.detection.base_detector import BaseDetector, Box, DetectorResult, clamp_box, expanded_box, safe_crop
from visimove.types import Frame


LEFT_EYE_INDICES = (33, 133, 159, 145, 153, 154, 155, 173, 246)
RIGHT_EYE_INDICES = (362, 263, 386, 374, 380, 381, 382, 398, 466)


@dataclass
class MediaPipeFaceMeshDetector:
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    reuse_frames: int = 2
    eye_padding_scale: float = 1.8
    face_landmarker_model_path: str | None = None
    _face_mesh: object | None = None
    _face_landmarker: object | None = None
    _api: str = ""
    _last_result: DetectorResult | None = None
    _reuse_count: int = 0

    def __post_init__(self) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("MediaPipe is not installed.") from exc

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._api = "classic"
            return

        self._face_landmarker = self._create_tasks_face_landmarker(mp)
        self._api = "tasks"

    def detect(self, frame: Frame, timestamp: float | None = None) -> DetectorResult:
        if self._last_result is not None and self._reuse_count < self.reuse_frames:
            self._reuse_count += 1
            return self._with_reused_crops(frame, self._last_result)

        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self._api == "classic":
            return self._detect_classic(frame, rgb, width, height)
        return self._detect_tasks(frame, rgb, width, height)

    def _detect_classic(self, frame: Frame, rgb: Frame, width: int, height: int) -> DetectorResult:
        result = self._face_mesh.process(rgb)  # type: ignore[union-attr]
        faces = getattr(result, "multi_face_landmarks", None)
        if not faces:
            self._last_result = None
            self._reuse_count = 0
            return DetectorResult(found=False)

        landmarks = faces[0].landmark
        pixel_landmarks = self._normalized_to_pixels(landmarks, width, height)
        return self._result_from_pixel_landmarks(frame, pixel_landmarks, "mediapipe_face_mesh")

    def _detect_tasks(self, frame: Frame, rgb: Frame, width: int, height: int) -> DetectorResult:
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._face_landmarker.detect(image)  # type: ignore[union-attr]
        faces = getattr(result, "face_landmarks", None)
        if not faces:
            self._last_result = None
            self._reuse_count = 0
            return DetectorResult(found=False)

        pixel_landmarks = self._normalized_to_pixels(faces[0], width, height)
        return self._result_from_pixel_landmarks(frame, pixel_landmarks, "mediapipe_tasks_face_landmarker")

    def _result_from_pixel_landmarks(
        self,
        frame: Frame,
        pixel_landmarks: list[tuple[int, int]],
        source: str,
    ) -> DetectorResult:
        height, width = frame.shape[:2]
        face_box = self._points_to_box(pixel_landmarks, width, height)
        left_eye_box = self._eye_box(pixel_landmarks, LEFT_EYE_INDICES, width, height)
        right_eye_box = self._eye_box(pixel_landmarks, RIGHT_EYE_INDICES, width, height)
        detection = DetectorResult(
            found=face_box is not None,
            face_box=face_box,
            left_eye_box=left_eye_box,
            right_eye_box=right_eye_box,
            left_eye_crop=safe_crop(frame, left_eye_box),
            right_eye_crop=safe_crop(frame, right_eye_box),
            landmarks={
                "source": source,
                "points": pixel_landmarks,
            },
            confidence=self.min_detection_confidence,
        )
        self._last_result = detection if detection.found else None
        self._reuse_count = 0
        return detection

    def _create_tasks_face_landmarker(self, mp: object) -> object:
        model_path = Path(self.face_landmarker_model_path or "")
        if not self.face_landmarker_model_path or not model_path.exists():
            version = getattr(mp, "__version__", "unknown")
            raise RuntimeError(
                "Installed MediaPipe does not expose the classic solutions.face_mesh API "
                f"(version: {version}), and no valid Tasks face_landmarker_model_path was configured. "
                "Expected a .task model such as models/detection/face_landmarker.task."
            )

        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
        except ImportError as exc:
            raise RuntimeError("MediaPipe Tasks FaceLandmarker API is unavailable.") from exc

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=self.min_detection_confidence,
            min_face_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def _eye_box(
        self,
        points: list[tuple[int, int]],
        indices: tuple[int, ...],
        width: int,
        height: int,
    ) -> Box | None:
        eye_points = [points[index] for index in indices if index < len(points)]
        base = self._points_to_box(eye_points, width, height)
        if base is None:
            return None
        return expanded_box(base, self.eye_padding_scale, width, height)

    @staticmethod
    def _points_to_box(points: list[tuple[int, int]], width: int, height: int) -> Box | None:
        if not points:
            return None
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return clamp_box((min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)), width, height)

    @staticmethod
    def _normalized_to_pixels(landmarks: object, width: int, height: int) -> list[tuple[int, int]]:
        return [
            (
                round(max(0.0, min(1.0, point.x)) * width),
                round(max(0.0, min(1.0, point.y)) * height),
            )
            for point in landmarks
        ]

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
