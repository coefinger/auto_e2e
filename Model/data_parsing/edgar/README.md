# Data parsing recipe for EDGAR rosbags

Turns TUM EDGAR rosbag2/mcap recordings into AutoE2E training shards, so the
existing `data_parsing.pre_extracted` loader, model and losses run on them
unchanged. Without `--map-dir` it produces the trajectory-imitation baseline
corpus; with it, every sample also carries the schema-v8 navigation members
(semantic BEV map + planned-route corridor) rendered from a lanelet2 map. No
reasoning labels, no world-model windows.

## Why a packer and not a `Dataset`

The other parsers (`kit_scenes`, `nvidia_physical_ai`, `l2d`) can index their
corpus randomly, so they subclass `torch.utils.data.Dataset`. An mcap file is a
sequential log: random access per `__getitem__` is not viable. EDGAR therefore
goes straight to the WebDataset shard contract that `make_pre_extracted_loader`
already reads, and training consumes shards like any other packed corpus.

## Two environments

Extraction needs ROS 2 (`rosbag2_py`, `rclpy`) and OpenCV; training needs torch.
They do not have to be the same interpreter — the shard directory is the only
thing passed between them.

```bash
# 1. extract (ROS 2 Humble python)
source /opt/ros/humble/setup.bash
python3 Model/data_parsing/edgar/extract_shards.py \
    --out /data/edgar_shards \
    /path/to/bag_a.mcap /path/to/multi_file_bag_dir

# 2. train (torch venv, from Model/)
python -m training.train_baseline --shard-dir /data/edgar_shards --epochs 10
```

Custom EDGAR/Autoware message packages are **not** needed: only
`sensor_msgs/CompressedImage` and `nav_msgs/Odometry` are deserialized. The
Autoware mode messages and `LaneletRoute` are read straight from CDR
(`_parse_mode`, `map_inputs.parse_lanelet_route`) because none of those
packages resolve outside alpamayo's container. The navigation path also needs
`pyproj`, `pyyaml`, `scipy` and the native rasterizer
(`python Model/navigation/native/build.py`, dependency-free C++17).

## Topics consumed

| Input | Topic |
| --- | --- |
| Cameras (6) | `/edgar/sensor/camera/basler/{view}/image_resized_compressed` |
| Egomotion | `/localization/kinematic_state`, falling back to `/edgar/sensor/gnss/novatel/center/odom` |
| Who drove (default on) | `/vehicle/status/control_mode`, `/control/vehicle_cmd_gate/operation_mode` — alpamayo's `driver_at` rule; only anchors whose full 12.8 s window the stack drove are packed (`--no-require-autonomous` to keep takeovers) |
| Planned route (`--map-dir`) | `/planning/mission_planning/route` (`LaneletRoute`), latest message at or before the anchor |

Default views are `frontleft, frontcenter, frontright, rearleft, rearcenter,
rearright` — a symmetric ring. `frontcenterreal` is omitted because it
duplicates `frontcenter`'s forward coverage, the same call the KITScenes parser
makes for its second front camera. Override with `--views`.

A bag missing the odometry topic or any requested camera is skipped with a
message rather than silently packed short.

A multi-file recording without `metadata.yaml` (the recorder was killed
mid-write) cannot be opened as one bag; `rebuild_metadata.py` regenerates it
from the chunks' own mcap summaries, dropping the unfinalized last chunk.

## Model inputs produced

- `cam_{i}.jpg` — `--image-size` (default 256) square JPEG per view
- `ego.npy` — float32 `[64×4 history | 64×2 target]`, time-major, the fixed
  AutoE2E ABI
- `meta.json` — `split_group_uid`, the recording this sample came from

### Egomotion history `(64, 4)`

Both odometry sources publish `nav_msgs/Odometry` with the twist in the body
frame, so the signals come off the message directly rather than by
differencing poses:

- `[0]` Speed (m/s) — `twist.linear.x`
- `[1]` Acceleration (m/s²) — **backward** difference of speed on the 10 Hz grid
- `[2]` Yaw rate (rad/s) — `twist.angular.z`
- `[3]` Curvature (1/m) — `yaw_rate / max(|speed|, 0.1)`

The backward difference is deliberate: a centered stencil would make the last
history row read one post-anchor sample, which is why the KITScenes policy has
to mask that value (`mask_latest_history_acceleration`). EDGAR is causal by
construction and needs no mask.

### Trajectory target `(64, 2)`

`[0]` acceleration (m/s²), `[1]` curvature (1/m) — the next 64 grid steps,
anchor excluded.

## Sampling

Odometry is resampled onto a 10 Hz grid spanning the bag. An anchor is valid
when 64 grid steps sit behind it and 64 ahead, all six views have produced a
frame, and the anchor speed is at least `--min-speed` (default 0.5 m/s) so
parked and queued segments do not dominate the corpus.

Camera frames are matched by **most recent frame at or before the anchor**, on
the bag receive clock. At the 20 Hz camera rate that is at most 50 ms of
staleness, and it keeps the whole camera pass to a single sequential read.

`split_group_uid` strips a trailing `_<n>` from the bag name, so `..._69.mcap`
and `..._70.mcap` — chunks of one recording — land in the same train/val group.
Without that, consecutive frames of one drive would straddle the split and the
holdout would measure memorization.

## Signal scales

`TrajectoryImitationLoss` divides each target signal by a corpus scale.
Extraction measures the per-signal RMS over everything it packed and writes it
to `manifest.json` as `measured_signal_scales`; `training/train_baseline.py`
builds its `DatasetTrainingPolicy` from that value rather than inheriting L2D's.

## Map and route (`--map-dir`)

```bash
python3 Model/data_parsing/edgar/extract_shards.py --out /data/edgar_nav --stride 0.5 \
    --map-dir /path/to/autoware.maps/lanelets/munich_map  /path/to/bags...
```

`map_inputs.py` reads the lanelet2 `.osm` directly (no `lanelet2` bindings) and
projects it with the map's own `map_projector_info.yaml` (TransverseMercator).
That *is* the bags' `map` frame: the lanelet IDs Autoware drew in
`/map/vector_map_marker` match the projected OSM to <2 m, so
`/localization/kinematic_state` poses are used as they are. Per anchor the
repo's `NativeNavigationRasterizer` renders the `kitscenes-v3-bev-1m-v1`
geometry (256×256 at 1 m/px, x ∈ [−85.5, 170.5] forward, ego at row 170):

- `map_semantic.npz` — 14 channels (drivable, lane boundary, centerline,
  intersection, crosswalk, stop line, traffic signal, direction sin/cos/valid,
  known area, …). Lanelets are oriented from geometry the way lanelet2 does
  (right bound aligned to left, pair flipped so left is left of travel);
  checked against the driven trace, 98% of moving poses agree with the nearest
  centerline's direction.
- `route_mask.npz` — corridor (3.5 m) of the planner's `preferred_primitive`
  per segment plus the destination marker; `route_valid=False` (and an
  explicit `no_route_yet`) before the first route message.
- `route_supervision.npz`, `navigation_meta.json` — as KITScenes.

`manifest.json` then says `map_context_channels: 14`, which the trainer and
`evaluation/eval_checkpoint.py` read to size the map encoder. A 14-channel
checkpoint cannot consume map-less shards (the loader's fallback is 3-channel),
so every corpus it is evaluated on needs `--map-dir` too.

Bags without Autoware localization (only the INS `.../novatel/center/odom`, in
UTM) need `--odom-utm-zone N`; positions are re-projected through lat/lon and
yaw is corrected for the grid-convergence difference (≈1.3° for Berlin on
`berlin_map`). This assumes that odometry's orientation is a heading in the
UTM grid — verify on the bag before trusting it.

`--survey` reports autonomous coverage per bag without touching cameras;
`--max-samples N` caps each bag for smoke runs of an otherwise hours-long job.

## Known limitations

- **No geometry.** `geometry_type` is `pseudo`; camera calibration is not read
  from the bag, so BEV fusion has no real projection to work with. Feature
  concat and cross-attention fusion are unaffected.
- **No map or route without `--map-dir`.** Then `map_valid` and `route_valid`
  are False for every sample and those branches are gated off.
- **Centerlines and intersections are approximate.** Centerline = mean of the
  arc-length-resampled bounds; intersection = `turn_direction` or a shared
  end-node fan-in/out, not lanelet2's routing graph. Both are at the 1 m/px
  raster's resolution.
- **Bag receive time, not header stamps**, for both cameras and odometry —
  consistent across streams, but it carries the recording latency.
