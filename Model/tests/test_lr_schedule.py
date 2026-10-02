"""The cosine schedule and the epoch-length count it depends on."""
import math

from training.train_baseline import cosine_with_warmup, train_steps_per_epoch

FLOOR = 1e-6 / 1e-4  # --lr-min / --lr


def test_warmup_rises_then_cosine_falls():
    warm, total = 100, 1000
    assert cosine_with_warmup(0, warm, total, FLOOR) == 0.01      # not zero: step 0 still trains
    assert cosine_with_warmup(warm - 1, warm, total, FLOOR) == 1.0  # peaks at base lr
    assert math.isclose(cosine_with_warmup(warm, warm, total, FLOOR), 1.0, rel_tol=1e-6)
    mid = cosine_with_warmup(warm + (total - warm) // 2, warm, total, FLOOR)
    assert math.isclose(mid, FLOOR + (1 - FLOOR) * 0.5, rel_tol=1e-3)
    assert math.isclose(cosine_with_warmup(total, warm, total, FLOOR), FLOOR, abs_tol=1e-9)


def test_monotone_after_warmup_and_clamped_past_the_end():
    warm, total = 50, 500
    vals = [cosine_with_warmup(s, warm, total, FLOOR) for s in range(warm, total + 1)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:])), "cosine must not rise"
    # The trailing partial-accumulation flush can push past total_steps; the lr
    # must stay at the floor rather than climbing back up the cosine.
    assert math.isclose(cosine_with_warmup(total + 37, warm, total, FLOOR), FLOOR, abs_tol=1e-9)


def test_no_warmup_starts_at_base_lr():
    assert cosine_with_warmup(0, 0, 100, FLOOR) == 1.0


def test_steps_per_epoch_excludes_the_holdout():
    manifest = {"total_samples": 100, "source_bags": [
        {"split_group_uid": "a", "samples": 60},
        {"split_group_uid": "b", "samples": 40}]}
    # 60 train samples, batch 16 -> 4 batches, accum 1 -> 4 steps
    assert train_steps_per_epoch(manifest, ["b"], 16, 1) == 4
    assert train_steps_per_epoch(manifest, ["b"], 16, 2) == 2   # accum halves the steps
    assert train_steps_per_epoch(manifest, [], 16, 1) == 7      # 100 samples -> 7 batches


def test_steps_per_epoch_falls_back_without_source_bags():
    assert train_steps_per_epoch({"total_samples": 32}, ["x"], 16, 1) == 2
    # Never zero: a scheduler built on 0 steps would divide by zero.
    assert train_steps_per_epoch({}, [], 16, 1) == 1
