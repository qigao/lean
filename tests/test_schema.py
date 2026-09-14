from yolo_flywire.schema import KeypointFrame, PointSequence


def test_point_sequence_rejects_inconsistent_keypoint_count():
    f0 = KeypointFrame(points=((0.0, 0.0, 1.0),))
    f1 = KeypointFrame(points=((0.0, 0.0, 1.0), (1.0, 1.0, 1.0)))
    try:
        PointSequence(frames=(f0, f1))
    except ValueError as exc:
        assert "keypoint count" in str(exc)
    else:
        raise AssertionError("expected ValueError")
