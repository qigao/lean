from yolo_flywire.synthetic import make_synthetic_dataset


def test_synthetic_dataset_is_seed_deterministic():
    a = make_synthetic_dataset(seed=7, samples_per_class=2, frames=12)
    b = make_synthetic_dataset(seed=7, samples_per_class=2, frames=12)
    assert a == b


def test_left_and_right_wave_have_opposite_wrist_displacement():
    data = make_synthetic_dataset(seed=1, samples_per_class=1, frames=12)
    by_label = {sample.label: sample for sample in data}
    left = by_label["wave_left"].sequence.frames
    right = by_label["wave_right"].sequence.frames
    assert left[-1].points[-1][0] - left[0].points[-1][0] < 0
    assert right[-1].points[-1][0] - right[0].points[-1][0] > 0
