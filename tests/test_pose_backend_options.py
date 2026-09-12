"""The pinned extractor uses the current precision API, not compatibility flags."""
from yolo_flywire.pose_backend import prediction_options


def test_prediction_options_use_explicit_cpu_float32_without_legacy_flags():
    options = prediction_options()
    assert options.get("quantize") == 32
    assert "half" not in options and "int8" not in options
    assert options["device"] == "cpu" and options["batch"] == 1
    assert options["max_det"] == 300 and options["classes"] == [0]
    assert options["imgsz"] == 640 and options["conf"] == 0.25 and options["iou"] == 0.7
    assert all(options[key] is False for key in (
        "rect", "augment", "save", "save_txt", "save_conf", "save_crop",
        "show", "visualize", "verbose", "stream",
    ))
