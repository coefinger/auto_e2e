#!/usr/bin/env python3
"""EDGAR rosbag -> AutoE2E WebDataset shards (imitation-only baseline).

Run with the ROS 2 Humble python (needs rosbag2_py + cv2), NOT the torch venv:

    source /opt/ros/humble/setup.bash
    python3 Model/data_parsing/edgar/extract_shards.py \
        --out /data/edgar_shards /path/to/bag.mcap /path/to/another_bag_dir

Writes exactly the shard contract ``data_parsing.pre_extracted`` already reads:

    <key>.cam_{i}.jpg   V camera views, image_size x image_size
    <key>.ego.npy       float32 [64*4 history | 64*2 target], time-major
    <key>.meta.json     split_group_uid = bag stem (train/val splits per bag)

plus a manifest.json for ``make_pre_extracted_loader``. No map/route/reasoning/
world-model members: this is the trajectory-imitation baseline.

Two passes per bag: odometry first (to build the 10 Hz grid and know which
anchors have a full 64-back/64-ahead window), then the camera topics streamed
once, holding the latest frame per view.

With ``--map-dir`` each sample additionally carries the schema-v8 navigation
members (``map_semantic.npz``, ``route_mask.npz``, ``route_supervision.npz``,
``navigation_meta.json``) rendered from the lanelet2 map and Autoware's planned
route at the anchor pose -- see ``map_inputs.py``.
"""

from __future__ import annotations

import argparse
import bisect
import io
import json
import re
import struct
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage

# Run as a script, sys.path[0] is this directory; the navigation package and
# data_parsing.edgar.map_inputs live under Model/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

HISTORY_STEPS = 64
FUTURE_STEPS = 64
HZ = 10.0
DT = 1.0 / HZ
MIN_SPEED_FOR_CURVATURE = 0.1  # m/s, matches the KITScenes parser

CAMERA_TOPIC = "/edgar/sensor/camera/basler/{view}/image_resized_compressed"
# frontcenterreal duplicates frontcenter's forward coverage; six views give a
# symmetric ring, same shape of input contract as the other parsers.
DEFAULT_VIEWS = (
    "frontleft", "frontcenter", "frontright",
    "rearleft", "rearcenter", "rearright",
)
# First match wins. kinematic_state is the fused localization twist in
# base_link; the novatel odom is the raw INS fallback.
ODOM_TOPICS = (
    "/localization/kinematic_state",
    "/edgar/sensor/gnss/novatel/center/odom",
)

# Who drove. Topics, tables and the driver_at rule are copied from alpamayo's
# scripts/eval_vs_autoware.py so a training sample and an eval stratum agree on
# who had the car. Both topics are optional: a bag without them can only be
# packed with --no-require-autonomous.
CONTROL_MODE_TOPIC = "/vehicle/status/control_mode"                 # autoware_vehicle_msgs/ControlModeReport
OPERATION_MODE_TOPIC = "/control/vehicle_cmd_gate/operation_mode"   # autoware_adapi_v1_msgs/OperationModeState
CONTROL_MODES = {0: "NO_COMMAND", 1: "AUTONOMOUS", 2: "AUTONOMOUS_STEER_ONLY",
                 3: "AUTONOMOUS_VELOCITY_ONLY", 4: "MANUAL", 5: "DISENGAGED",
                 6: "NOT_READY"}
OPERATION_MODES = {0: "UNKNOWN", 1: "STOP", 2: "AUTONOMOUS", 3: "LOCAL", 4: "REMOTE"}
# Autoware's planned route, one message per (re)plan. Read raw: the type does
# not resolve outside alpamayo's container; map_inputs parses it from CDR.
ROUTE_TOPIC = "/planning/mission_planning/route"                    # autoware_planning_msgs/LaneletRoute


def _parse_mode(raw: bytes) -> tuple[int, int]:
    """(own stamp ns, mode) from a CDR message that opens with Time + uint8 mode.

    Both mode messages have that prefix, and neither autoware_vehicle_msgs nor
    autoware_adapi_v1_msgs resolves outside alpamayo's container, so the two
    fields are read straight out of the buffer: 4-byte encapsulation header,
    int32 sec, uint32 nanosec, uint8 mode.
    """
    sec, nanosec = struct.unpack_from("<iI", raw, 4)
    return sec * 10 ** 9 + nanosec, raw[12]


def driver_at(ctrl_mode: str | None, op_mode: str | None) -> str:
    """alpamayo autocut.py's rule. The operation mode names the session and is
    asked first: a LOCAL/REMOTE operator drives with drive-by-wire engaged, so
    the control mode reads AUTONOMOUS throughout. Then any control mode but full
    AUTONOMOUS is a person in the seat -- the partial modes included, since a
    driver correcting the wheel while the computer holds speed has taken over."""
    if op_mode in ("LOCAL", "REMOTE"):
        return "remote"
    if ctrl_mode is not None and ctrl_mode != "AUTONOMOUS":
        return "human"
    return "stack"


def split_group(bag: Path) -> str:
    """Chunks of one recording ("..._69.mcap", "..._70.mcap") share a split
    group, or consecutive frames of the same drive straddle train/val."""
    return re.sub(r"_\d+$", "", bag.stem)


def _open(bag: Path):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    return reader


def _topics(bag: Path) -> set[str]:
    return {t.name for t in _open(bag).get_all_topics_and_types()}


def read_odometry_and_modes(bag: Path, odom_topic: str, available: set[str]):
    """One pass over the three small topics: odometry plus who drove.

    Returns (t_seconds, speed, yaw_rate, ctrl, op, pose, routes): ctrl/op are
    sorted [(t_ns, mode name)], pose is (N, 3) [x, y, yaw] in the odometry
    frame, routes is [(received_ns, raw cdr)]. Reading them together matters --
    on these bags a single filtered pass costs ~80 s, and doing odometry and
    modes separately would pay that twice per bag.
    """
    topics = [odom_topic] + [t for t in (CONTROL_MODE_TOPIC, OPERATION_MODE_TOPIC, ROUTE_TOPIC)
                             if t in available]
    reader = _open(bag)
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
    t, v, w, pose = [], [], [], []
    ctrl, op, routes = [], [], []
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic == odom_topic:
            msg = deserialize_message(data, Odometry)
            t.append(stamp * 1e-9)
            v.append(msg.twist.twist.linear.x)
            w.append(msg.twist.twist.angular.z)
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            pose.append((p.x, p.y, yaw_from_quaternion(q.x, q.y, q.z, q.w)))
        elif topic == ROUTE_TOPIC:
            routes.append((stamp, bytes(data)))
        elif topic == CONTROL_MODE_TOPIC:
            # Timed by its own stamp, as alpamayo does, so takeover instants
            # match the eval clips to the microsecond.
            own, mode = _parse_mode(data)
            ctrl.append((own, CONTROL_MODES.get(mode, f"UNKNOWN_{mode}")))
        else:
            # OperationModeState's own stamp is when the mode was SET (it reads
            # 0 on these bags and is republished unchanged), so use bag receipt.
            op.append((stamp, OPERATION_MODES.get(_parse_mode(data)[1],
                                                  f"UNKNOWN_{_parse_mode(data)[1]}")))
    ctrl.sort()
    op.sort()
    return (np.asarray(t), np.asarray(v), np.asarray(w), ctrl, op,
            np.asarray(pose, dtype=np.float64).reshape(-1, 3), routes)


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Heading about +z, CCW from +x, in the odometry frame."""
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def pose_grid(t, pose, grid):
    """Resample [x, y, yaw] onto the 10 Hz grid; yaw is unwrapped first so the
    interpolation never averages across the +-pi seam."""
    x = np.interp(grid, t, pose[:, 0])
    y = np.interp(grid, t, pose[:, 1])
    yaw = np.interp(grid, t, np.unwrap(pose[:, 2]))
    yaw = (yaw + np.pi) % (2.0 * np.pi) - np.pi
    return np.stack([x, y, yaw], axis=1)


def non_stack_intervals(ctrl, op, t_start_ns: int):
    """Merged [start, end) spans (ns) in which anyone but the stack drove.

    Exact rather than sampled: the driver can only change at a message of either
    topic, so the step function is evaluated once per edge. Time before the
    first control-mode message is unknown, hence not-stack -- the same call
    alpamayo's driver_over makes when it returns None.
    """
    inf = float("inf")
    if not ctrl:
        return [(t_start_ns, inf)]
    bad = []
    if ctrl[0][0] > t_start_ns:
        bad.append((t_start_ns, ctrl[0][0]))
    ct = [t for t, _ in ctrl]
    ot = [t for t, _ in op]
    edges = sorted({*ct, *ot})
    for i, edge in enumerate(edges):
        nxt = edges[i + 1] if i + 1 < len(edges) else inf
        ci = bisect.bisect_right(ct, edge) - 1
        oi = bisect.bisect_right(ot, edge) - 1
        if ci < 0 or driver_at(ctrl[ci][1], op[oi][1] if oi >= 0 else None) != "stack":
            bad.append((edge, nxt))
    merged = []
    for start, end in bad:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [tuple(x) for x in merged]


def stack_anchor_mask(grid, first: int, last: int, bad) -> np.ndarray:
    """Which anchors in [first, last) have the stack driving the whole window.

    The window is the full 12.8 s the sample covers, history included: the past
    is model input, so a takeover anywhere in it makes the sample part-human.
    """
    idx = np.arange(first, last)
    mask = np.ones(len(idx), dtype=bool)
    if not bad:
        return mask
    starts = np.array([b[0] for b in bad], dtype=np.float64)
    ends = np.array([b[1] for b in bad], dtype=np.float64)
    win_start = grid[idx - HISTORY_STEPS] * 1e9
    win_end = grid[idx + FUTURE_STEPS] * 1e9
    # Intervals are disjoint and sorted, so only the last one starting before
    # the window ends can overlap it. side="left": a span opening exactly as the
    # window closes touches it, it does not overlap it.
    j = np.searchsorted(starts, win_end, side="left") - 1
    hit = j >= 0
    mask[hit] = ends[j[hit]] <= win_start[hit]
    return mask


def egomotion_grid(t, v, w):
    """Resample onto the 10 Hz grid and derive the four egomotion signals."""
    grid = np.arange(t[0], t[-1], DT)
    speed = np.interp(grid, t, v)
    yaw_rate = np.interp(grid, t, w)
    # Backward difference keeps acceleration causal: no history row reads a
    # future pose, so nothing has to be masked at train time.
    accel = np.concatenate([[0.0], np.diff(speed) / DT])
    curvature = yaw_rate / np.maximum(np.abs(speed), MIN_SPEED_FOR_CURVATURE)
    return grid, np.stack([speed, accel, yaw_rate, curvature], axis=1)


def to_jpeg(raw: bytes, size: int) -> bytes:
    """Transcode one CompressedImage payload to a size x size JPEG.

    imdecode gives BGR and imencode consumes BGR, so the round trip preserves
    the original channel order — no cvtColor needed.
    """
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("undecodable camera frame")
    frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("jpeg encode failed")
    return buf.tobytes()


class ShardWriter:
    """Rolling .tar writer; one shard per ``samples_per_shard`` samples."""

    def __init__(self, out_dir: Path, samples_per_shard: int):
        self.out_dir = out_dir
        self.samples_per_shard = samples_per_shard
        self.tar = None
        self.shard_index = 0
        self.in_shard = 0
        self.total = 0

    def _member(self, name: str, payload: bytes) -> None:
        info = tarfile.TarInfo(name=name)
        info.size = len(payload)
        self.tar.addfile(info, io.BytesIO(payload))

    def write(self, key: str, jpegs: list[bytes], ego: np.ndarray, group: str,
              extra: dict[str, bytes] | None = None):
        if self.tar is None or self.in_shard >= self.samples_per_shard:
            self.close()
            path = self.out_dir / f"edgar-{self.shard_index:05d}.tar"
            self.tar = tarfile.open(path, "w")
            self.shard_index += 1
            self.in_shard = 0
        for i, jpeg in enumerate(jpegs):
            self._member(f"{key}.cam_{i}.jpg", jpeg)
        self._member(f"{key}.ego.npy", ego.astype(np.float32).tobytes())
        self._member(f"{key}.meta.json", json.dumps({"split_group_uid": group}).encode())
        for name, payload in (extra or {}).items():
            self._member(f"{key}.{name}", payload)
        self.in_shard += 1
        self.total += 1

    def close(self):
        if self.tar is not None:
            self.tar.close()
            self.tar = None


def extract_bag(bag: Path, views, writer: ShardWriter | None, size: int,
                min_speed: float, require_autonomous: bool = True,
                stride: float = DT, navigation=None, max_samples: int = 0,
                odom_utm_zone: int | None = None):
    """Stream one bag into the shard writer. Returns (samples, target rows).

    ``writer=None`` surveys: everything up to the camera pass runs, nothing is
    written. That is the cheap half (~80 s/bag against ~40 min), so it answers
    "how much autonomous driving is in here" without decoding a frame.
    """
    available = _topics(bag)
    odom_topic = next((t for t in ODOM_TOPICS if t in available), None)
    if odom_topic is None:
        print(f"  skip {bag.name}: no odometry topic ({'/'.join(ODOM_TOPICS)})")
        return 0, np.zeros((0, 2))
    cam_topics = [CAMERA_TOPIC.format(view=v) for v in views]
    missing = [t for t in cam_topics if t not in available]
    if missing:
        print(f"  skip {bag.name}: missing camera topics {missing}")
        return 0, np.zeros((0, 2))
    if require_autonomous and CONTROL_MODE_TOPIC not in available:
        print(f"  skip {bag.name}: no {CONTROL_MODE_TOPIC}; who drove is unknown "
              f"(pass --no-require-autonomous to pack it anyway)")
        return 0, np.zeros((0, 2))

    t, v, w, ctrl, op, pose, routes = read_odometry_and_modes(bag, odom_topic, available)
    if len(t) < 2:
        print(f"  skip {bag.name}: {len(t)} odometry messages")
        return 0, np.zeros((0, 2))
    grid, signals = egomotion_grid(t, v, w)
    first = HISTORY_STEPS
    last = len(grid) - FUTURE_STEPS  # exclusive
    if last <= first:
        print(f"  skip {bag.name}: {len(grid)} grid steps, need >{HISTORY_STEPS + FUTURE_STEPS}")
        return 0, np.zeros((0, 2))
    print(f"  {bag.name}: {odom_topic}, {grid[-1] - grid[0]:.0f}s, "
          f"{last - first} candidate anchors, median speed {np.median(np.abs(v)):.1f} m/s")

    moving = np.abs(signals[first:last, 0]) >= min_speed
    if require_autonomous:
        bad = non_stack_intervals(ctrl, op, int(grid[0] * 1e9))
        stack_mask = stack_anchor_mask(grid, first, last, bad)
        print(f"    ctrl {len(ctrl)} msgs, op {len(op)} msgs, "
              f"{len(bad)} non-stack spans; "
              f"{stack_mask.sum()}/{len(stack_mask)} anchors fully autonomous, "
              f"{int((stack_mask & moving).sum())} also moving")
    else:
        stack_mask = np.ones(last - first, dtype=bool)
    # Anchors are 0.1 s apart, so neighbours are near-duplicates: 4 h of driving
    # at 10 Hz is 150k samples that mostly repeat each other. Thinning trades
    # redundancy for epochs that finish.
    step = max(1, int(round(stride / DT)))
    if step > 1:
        stack_mask = stack_mask & (((np.arange(first, last) - first) % step) == 0)
    expected = int((stack_mask & moving).sum())
    if not stack_mask.any():
        print(f"  skip {bag.name}: no fully-autonomous anchor")
        return 0, np.zeros((0, 2))
    if writer is None:
        if step > 1:
            print(f"    at stride {stride:g}s -> {expected} samples")
        return expected, np.zeros((0, 2))

    bag_nav = None
    poses = None
    if navigation is not None:
        if odom_topic != ODOM_TOPICS[0] and odom_utm_zone is None:
            # Only the fused localization pose is in the lanelet map's frame; the
            # raw INS odom is UTM and needs --odom-utm-zone to be projected.
            print(f"  skip {bag.name}: navigation needs {ODOM_TOPICS[0]} (map frame), "
                  f"bag has {odom_topic}; pass --odom-utm-zone to project it")
            return 0, np.zeros((0, 2))
        if odom_topic != ODOM_TOPICS[0]:
            pose = navigation.utm_to_map(pose, odom_utm_zone)
            print(f"    poses: {odom_topic} projected from UTM {odom_utm_zone} into "
                  f"{navigation.lmap.name}'s frame")
        poses = pose_grid(t, pose, grid)
        bag_nav = navigation.for_bag(bag.name, routes, poses[first:last, :2])
        print(f"    map: {bag_nav.stats['map_primitives']}; routes: "
              f"{bag_nav.stats['routes_valid']}/{bag_nav.stats['routes']} valid")

    reader = _open(bag)
    reader.set_filter(rosbag2_py.StorageFilter(topics=cam_topics))
    latest: dict[str, bytes] = {}
    anchor = first
    written = 0
    targets = []
    group = split_group(bag)

    def emit(idx: int) -> None:
        nonlocal written
        if len(latest) < len(cam_topics):
            return
        if not stack_mask[idx - first]:
            return
        if abs(signals[idx, 0]) < min_speed:
            return
        history = signals[idx - HISTORY_STEPS:idx]                       # (64, 4)
        target = signals[idx + 1:idx + 1 + FUTURE_STEPS][:, [1, 3]]      # (64, 2)
        try:
            jpegs = [to_jpeg(latest[t], size) for t in cam_topics]
        except ValueError as error:
            print(f"    anchor {idx}: {error}")
            return
        ego = np.concatenate([history.reshape(-1), target.reshape(-1)])
        extra = None
        if bag_nav is not None:
            from navigation.rasterizer import EgoPose
            extra = bag_nav.members(EgoPose(float(poses[idx, 0]), float(poses[idx, 1]),
                                            float(poses[idx, 2]), int(round(grid[idx] * 1e9))))
        writer.write(f"{group}_{idx:07d}", jpegs, ego, group, extra)
        targets.append(target)
        written += 1

    while reader.has_next():
        topic, data, stamp = reader.read_next()
        now = stamp * 1e-9
        while anchor < last and now >= grid[anchor]:
            emit(anchor)
            anchor += 1
        if anchor >= last or (max_samples and written >= max_samples):
            break
        latest[topic] = deserialize_message(data, CompressedImage).data.tobytes()

    print(f"  {bag.name}: wrote {written} samples")
    return written, (np.concatenate(targets) if targets else np.zeros((0, 2)))


def self_check() -> None:
    """Signal derivation on a synthetic constant-curvature arc."""
    radius, speed, ramp = 50.0, 10.0, 0.5
    t = np.arange(0.0, 20.0, 0.02)
    v = speed + ramp * t
    w = v / radius
    grid, signals = egomotion_grid(t, v, w)

    assert np.allclose(np.diff(grid), DT), "grid is not 10 Hz"
    interior = slice(1, -1)
    assert np.allclose(signals[interior, 0], speed + ramp * (grid[interior] - t[0]),
                       atol=1e-6), "speed does not track the odometry"
    assert np.allclose(signals[interior, 1], ramp, atol=1e-6), "acceleration is wrong"
    assert np.allclose(signals[interior, 3], 1.0 / radius, atol=1e-9), "curvature is wrong"
    # Backward difference: no row may read a future sample. Perturbing only the
    # last grid step must leave every earlier acceleration untouched.
    bumped = v.copy()
    bumped[t >= grid[-1]] += 5.0
    _, perturbed = egomotion_grid(t, bumped, w)
    assert np.allclose(signals[:-1, 1], perturbed[:-1, 1]), "acceleration is not causal"
    # Who drove, and which anchors survive it.
    s = 10 ** 9
    ctrl = [(0, "AUTONOMOUS"), (30 * s, "MANUAL"), (40 * s, "AUTONOMOUS")]
    op = [(0, "AUTONOMOUS")]
    bad = non_stack_intervals(ctrl, op, 0)
    assert bad == [(30 * s, 40 * s)], bad
    # A REMOTE operation mode is not the stack even while control reads AUTONOMOUS.
    assert non_stack_intervals([(0, "AUTONOMOUS")], [(0, "REMOTE")], 0) == [(0, float("inf"))]
    # Partial autonomy is a person in the seat.
    assert driver_at("AUTONOMOUS_STEER_ONLY", "AUTONOMOUS") == "human"
    assert driver_at("AUTONOMOUS", "AUTONOMOUS") == "stack"
    # Unknown before the first control-mode message.
    assert non_stack_intervals([(5 * s, "AUTONOMOUS")], [], 0) == [(0, 5 * s)]

    # 100 s grid, takeover at 30-40 s: every window touching it must be dropped,
    # and no window that merely abuts it may be.
    grid = np.arange(0.0, 100.0, DT)
    first, last = HISTORY_STEPS, len(grid) - FUTURE_STEPS
    mask = stack_anchor_mask(grid, first, last, bad)
    kept = np.arange(first, last)[mask]
    assert kept.size, "everything dropped"
    for idx in kept:
        assert grid[idx + FUTURE_STEPS] <= 30.0 or grid[idx - HISTORY_STEPS] >= 40.0
    dropped = set(range(first, last)) - set(kept.tolist())
    for idx in dropped:
        assert grid[idx - HISTORY_STEPS] < 40.0 and grid[idx + FUTURE_STEPS] > 30.0
    print("[edgar] self-check passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bags", nargs="*", help=".mcap files or rosbag2 directories")
    ap.add_argument("--self-check", action="store_true",
                    help="verify the signal derivation and exit")
    ap.add_argument("--out", help="output shard directory")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--views", nargs="+", default=list(DEFAULT_VIEWS))
    ap.add_argument("--samples-per-shard", type=int, default=200)
    ap.add_argument("--stride", type=float, default=DT,
                    help="seconds between anchors (default %(default)s = every "
                         "10 Hz step). Raise it on long recordings: neighbouring "
                         "anchors overlap in 12.7 of their 12.8 s")
    ap.add_argument("--survey", action="store_true",
                    help="report autonomous coverage per bag and exit; reads "
                         "only odometry and the mode topics, no camera decode")
    ap.add_argument("--no-require-autonomous", dest="require_autonomous",
                    action="store_false",
                    help="keep anchors the safety driver drove. Off by default: "
                         "these are disengagement loggings, and imitating a "
                         "takeover teaches the policy the human's correction")
    ap.add_argument("--map-dir",
                    help="lanelet2 map directory (lanelet2_map.osm + "
                         "map_projector_info.yaml) in the bags' map frame. With it, "
                         "every sample also carries the schema-v8 navigation members "
                         "(14-channel semantic map, route corridor, route supervision)")
    ap.add_argument("--odom-utm-zone", type=int,
                    help="the INS odometry fallback (novatel .../odom) is in this UTM "
                         "zone; needed with --map-dir on bags without Autoware "
                         "localization (Berlin resim: 33). Assumes its orientation "
                         "is a heading in that grid, CCW from grid east")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="stop each bag after this many samples; a smoke-test knob "
                         "for a pipeline that otherwise runs for hours (0 = all)")
    ap.add_argument("--min-speed", type=float, default=0.5,
                    help="drop anchors below this speed (m/s) so parked "
                         "segments do not dominate the corpus")
    args = ap.parse_args()
    if args.self_check:
        self_check()
        return
    if not args.bags or (not args.out and not args.survey):
        ap.error("packing needs at least one bag and --out")

    navigation = None
    if args.map_dir and not args.survey:
        from data_parsing.edgar.map_inputs import EdgarNavigation
        navigation = EdgarNavigation(args.map_dir)
        print(f"[edgar] map {navigation.lmap.name}: {len(navigation.lmap.lanelets)} lanelets, "
              f"{len(navigation.lmap.stop_lines)} stop lines, {len(navigation.lmap.signals)} "
              f"signals; raster {navigation.geometry_id}")
    writer = None
    if not args.survey:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        writer = ShardWriter(out_dir, args.samples_per_shard)
    all_targets = []
    groups = []
    sources: list[dict[str, object]] = []
    for bag in (Path(b) for b in args.bags):
        print(f"[edgar] {bag}")
        written, targets = extract_bag(bag, args.views, writer, args.image_size,
                                       args.min_speed, args.require_autonomous,
                                       args.stride, navigation, args.max_samples,
                                       args.odom_utm_zone)
        # Record every bag offered, kept or not, so the corpus explains itself
        # without going back to this run's stdout.
        sources.append({"bag": str(bag.resolve()),
                        "split_group_uid": split_group(bag),
                        "samples": written})
        if written:
            groups.append(split_group(bag))
            all_targets.append(targets)
    if writer is None:
        total = sum(int(x["samples"]) for x in sources)
        print(f"\n[edgar] survey: {total} autonomous anchors across {len(sources)} bags")
        for x in sources:
            print(f"  {x['samples']:7d}  {x['split_group_uid']}")
        return
    writer.close()

    if not writer.total:
        raise SystemExit("no samples extracted")
    targets = np.concatenate(all_targets)
    # RMS per signal is what TrajectoryImitationLoss divides by, so report it
    # here and put it in the manifest rather than inheriting L2D's scales.
    scales = np.sqrt((targets ** 2).mean(axis=0))
    manifest = {
        "dataset": "TUM-EDGAR/rosbags",
        "total_samples": writer.total,
        "shards": writer.shard_index,
        "num_views": len(args.views),
        "views": list(args.views),
        "image_size": args.image_size,
        "has_map": navigation is not None,
        "geometry_type": "pseudo",
        # 14 semantic channels on the navigation path, RGB-placeholder 3 without.
        "map_context_channels": 14 if navigation is not None else 3,
        "route_channels": 2,
        "navigation": None if navigation is None else {
            "map_dir": str(Path(args.map_dir).resolve()),
            "map_name": navigation.lmap.name,
            "map_sha256": navigation.lmap.map_sha256,
            "geometry_id": navigation.geometry_id,
            "route_topic": ROUTE_TOPIC,
        },
        "split_groups": sorted(set(groups)),
        "source_bags": sources,
        "measured_signal_scales": [float(scales[0]), float(scales[1])],
        "min_speed_m_s": args.min_speed,
        "autonomous_only": args.require_autonomous,
        "anchor_stride_s": args.stride,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[edgar] {writer.total} samples in {writer.shard_index} shards -> {out_dir}")
    print(f"[edgar] measured signal RMS: accel={scales[0]:.4f} curvature={scales[1]:.4f}")
    print(f"[edgar] split groups: {len(set(groups))}")


if __name__ == "__main__":
    main()
