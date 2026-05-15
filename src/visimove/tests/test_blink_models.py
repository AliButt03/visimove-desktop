from __future__ import annotations

import numpy as np

from visimove.blink import KeyboardDummyBlinkModel, OnnxBlinkModel, OnnxBlinkModelConfig
from visimove.blink.base_blink_model import empty_blink_result
from visimove.detection import DummyDetector
from visimove.pipeline.realtime_pipeline import build_blink_model


def test_dummy_blink_model_returns_probability_result() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    detection = DummyDetector().detect(frame)

    result = KeyboardDummyBlinkModel(fake_probability=0.12).infer(frame, detection)

    assert result.left_closed_probability == 0.12
    assert result.right_closed_probability == 0.12
    assert result.combined_closed_probability == 0.12
    assert result.confidence == 1.0


def test_missing_onnx_model_falls_back_to_dummy() -> None:
    model = build_blink_model({"backend": "onnx", "model_path": "models/blink/missing.onnx"})

    assert isinstance(model, KeyboardDummyBlinkModel)


def test_onnx_preprocess_eye_crop_resizes_and_normalizes() -> None:
    model = object.__new__(OnnxBlinkModel)
    model.config = OnnxBlinkModelConfig(model_path="unused.onnx", input_width=32, input_height=24)
    crop = np.full((10, 20, 3), 255, dtype=np.uint8)

    tensor = model.preprocess_eye_crop(crop)

    assert tensor.shape == (1, 3, 24, 32)
    assert tensor.dtype == np.float32
    assert float(tensor.max()) == 1.0


def test_empty_blink_result_is_safe_for_missing_crops() -> None:
    result = empty_blink_result()

    assert result.combined_closed_probability == 0.0
    assert result.confidence == 0.0
    assert result.inference_time_ms == 0.0
