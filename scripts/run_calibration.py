from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visimove.calibration.calibration_ui import CalibrationUi, CalibrationUiConfig
from visimove.calibration.eyetrax_provider import EyeTraxCalibrationGazeProvider
from visimove.calibration.mobilegaze_provider import MobileGazeCalibrationGazeProvider
from visimove.config import load_config
from visimove.gaze.eyetrax_adapter import EyeTraxAdapter
from visimove.gaze.mobilegaze_adapter import MobileGazeAdapter
from visimove.pipeline.realtime_pipeline import build_detector


def resolve_mapping_settings(config: dict[str, object], gaze_backend: str) -> tuple[str, str]:
    mapping_model_type = str(config.get("mapping_model", config.get("mapping_model_type", "affine")))
    mapping_fit_strategy = str(config.get("mapping_fit_strategy", "point_means"))
    if gaze_backend == "mobilegaze":
        mapping_model_type = str(config.get("mobilegaze_mapping_model", "auto"))
        mapping_fit_strategy = str(config.get("mobilegaze_mapping_fit_strategy", "point_means"))
    return mapping_model_type, mapping_fit_strategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VisiMove screen calibration.")
    parser.add_argument("--config", default="config/calibration.yaml")
    parser.add_argument("--mode", choices=["5", "9", "16"], default=None)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--gaze-backend", choices=("dummy", "eyetrax", "gazefollower", "mobilegaze"))
    parser.add_argument("--detector-backend", choices=("dummy", "opencv", "mediapipe", "yolo"))
    parser.add_argument("--blink-backend", choices=("dummy", "ocec", "onnx"))
    parser.add_argument("--verbose-quality", action="store_true", help="Print detailed calibration quality diagnostics.")
    args = parser.parse_args()

    full_config = load_config("config/default.yaml", args.config)
    config = full_config.get("calibration", {})
    gaze_config = full_config.get("gaze", {})
    detection_config = full_config.get("detection", {})
    blink_config = full_config.get("blink", {})

    gaze_backend = args.gaze_backend or str(gaze_config.get("gaze_backend", gaze_config.get("backend", "dummy")))
    detector_backend = args.detector_backend or str(
        detection_config.get("detector_backend", detection_config.get("backend", "auto"))
    )
    blink_backend = args.blink_backend or str(blink_config.get("backend", "dummy"))

    if gaze_backend == "eyetrax":
        eyetrax_config = full_config.get("eyetrax", {})
        eyetrax_model_path = gaze_config.get("model_path") or eyetrax_config.get("model_path", "models/gaze/eyetrax")
        eyetrax_face_model_path = eyetrax_config.get(
            "face_landmarker_model_path",
            "models/detection/face_landmarker.task",
        )
        status = EyeTraxAdapter.check_setup(
            repo_path=eyetrax_config.get("repo_path", "external/eyetrax"),
            model_path=eyetrax_model_path,
            face_landmarker_model_path=eyetrax_face_model_path,
        )
        if not status.ready:
            print(f"EyeTrax calibration refused: {status.reason}")
            print(
                "No calibration profile was written. Set up EyeTrax, create/download a real "
                "EyeTrax gaze model, and provide a local face_landmarker.task first."
            )
            raise SystemExit(1)
        eyetrax_adapter = EyeTraxAdapter(
            repo_path=eyetrax_config.get("repo_path", "external/eyetrax"),
            model_path=eyetrax_model_path,
            face_landmarker_model_path=eyetrax_face_model_path,
            use_gpu=bool(eyetrax_config.get("use_gpu", False)),
            input_size=eyetrax_config.get("input_size"),
        )
        provider = EyeTraxCalibrationGazeProvider(eyetrax_adapter, args.camera_index)
    elif gaze_backend == "mobilegaze":
        mobilegaze_config = full_config.get("mobilegaze", {})
        mobilegaze_model_path = gaze_config.get("model_path") or mobilegaze_config.get(
            "model_path",
            "external/mobilegaze/weights/mobileone_s0_gaze.onnx",
        )
        status = MobileGazeAdapter.check_setup(
            repo_path=mobilegaze_config.get("repo_path", "external/mobilegaze"),
            model_path=mobilegaze_model_path,
        )
        if not status.ready:
            print(f"MobileGaze calibration refused: {status.reason}")
            print(
                "No calibration profile was written. Set up MobileGaze and provide a local "
                "ONNX model before calibrating."
            )
            raise SystemExit(1)
        mobilegaze_adapter = MobileGazeAdapter.from_config(
            {
                **mobilegaze_config,
                "model_path": mobilegaze_model_path,
            }
        )
        detector_config = {
            **detection_config,
            "detector_backend": detector_backend,
            "backend": detector_backend,
        }
        provider = MobileGazeCalibrationGazeProvider(
            mobilegaze_adapter,
            build_detector(detector_config),
            args.camera_index,
        )
    else:
        provider = None

    output_path = args.output
    if output_path is None and gaze_backend == "eyetrax":
        output_path = "data/calibration/user_profile_eyetrax.json"
    if output_path is None and gaze_backend == "mobilegaze":
        output_path = "data/calibration/user_profile_mobilegaze.json"
    if output_path is None:
        output_path = str(config.get("output_path", config.get("profile_path", "data/calibration/user_profile.json")))

    eyetrax_metadata = {}
    if gaze_backend == "eyetrax":
        eyetrax_config = full_config.get("eyetrax", {})
        eyetrax_metadata = {
            "eyetrax_model_path": str(eyetrax_model_path),
            "eyetrax_repo_path": str(eyetrax_config.get("repo_path", "external/eyetrax")),
            "face_landmarker_model_path": str(eyetrax_face_model_path),
        }
    mobilegaze_metadata = {}
    if gaze_backend == "mobilegaze":
        mobilegaze_config = full_config.get("mobilegaze", {})
        mobilegaze_metadata = {
            "mobilegaze_model_path": str(mobilegaze_model_path),
            "mobilegaze_repo_path": str(mobilegaze_config.get("repo_path", "external/mobilegaze")),
            "mobilegaze_preprocessing_version": MobileGazeAdapter.preprocessing_version,
            "mobilegaze_temporal_median_window": str(
                mobilegaze_adapter._raw_gaze_filter.window_size
            ),
            "native_output_units": "radians_yaw_pitch",
        }
    mapping_model_type, mapping_fit_strategy = resolve_mapping_settings(config, gaze_backend)

    ui_config = CalibrationUiConfig(
        mode=args.mode or str(config.get("mode", "9")),
        camera_index=args.camera_index,
        output_path=output_path,
        mapping_model_type=mapping_model_type,
        mapping_fit_strategy=mapping_fit_strategy,
        stabilization_seconds=float(config.get("stabilization_seconds", 0.75)),
        sample_seconds=float(config.get("sample_seconds", 1.7)),
        stabilization_ms=int(config.get("stabilization_ms", 800)),
        collection_ms=int(config.get("collection_ms", 2500)),
        min_valid_samples_per_point=int(config.get("min_valid_samples_per_point", 15)),
        max_collection_ms_per_point=int(config.get("max_collection_ms_per_point", 4500)),
        sample_interval_ms=int(config.get("sample_interval_ms", 0)),
        gaze_backend=gaze_backend,
        detector_backend=detector_backend,
        blink_backend=blink_backend,
        backend_metadata={
            "gaze_model_path": str(gaze_config.get("model_path")),
            "blink_model_path": str(blink_config.get("model_path")),
            **eyetrax_metadata,
            **mobilegaze_metadata,
        },
        min_confidence=float(gaze_config.get("confidence_threshold", 0.5)),
        verbose_quality=bool(args.verbose_quality),
    )
    print(
        "Calibration settings: "
        f"layout={ui_config.mode}-point, gaze_backend={gaze_backend}, "
        f"detector_backend={detector_backend}, blink_backend={blink_backend}, "
        f"output={ui_config.output_path}"
    )
    if gaze_backend == "dummy":
        print(
            "Calibration warning: dummy gaze calibration is only for pipeline testing and "
            "is not valid for real gaze backends."
        )
    if gaze_backend == "eyetrax":
        print(
            "EyeTrax calibration: collecting real EyeTrax output. Look at each dot until it advances. "
            "Do not enable cursor until calibration quality is good."
        )
    if gaze_backend == "mobilegaze":
        print(
            "MobileGaze calibration: collecting real MobileGaze yaw/pitch output. Look at each dot "
            "until it advances. Do not enable cursor until calibration quality is good or acceptable."
        )
    CalibrationUi(ui_config, provider=provider).run()


if __name__ == "__main__":
    main()
