"""EDGAR map/route inputs: CDR parsing, bound orientation, route assembly."""
import struct

import numpy as np
import pytest

from data_parsing.edgar.map_inputs import (
    Lanelet,
    LaneletXmlMap,
    orient_bounds,
    parse_lanelet_route,
    yaw_from_quaternion,
)
from navigation.contracts import MapFrame, Maneuver, TransitionType


# ------------------------------------------------------------------- CDR encode
class _CdrWriter:
    """Mirror of the reader, to build a LaneletRoute the way rmw_cyclonedds would."""

    def __init__(self):
        self.b = bytearray(b"\x00\x01\x00\x00")  # CDR_LE encapsulation

    def _align(self, n):
        self.b += b"\x00" * ((-(len(self.b) - 4)) % n)

    def _put(self, fmt, size, v):
        self._align(size)
        self.b += struct.pack(fmt, v)

    def u32(self, v):
        self._put("<I", 4, v)

    def i32(self, v):
        self._put("<i", 4, v)

    def i64(self, v):
        self._put("<q", 8, v)

    def f64(self, v):
        self._put("<d", 8, v)

    def u8(self, v):
        self.b += bytes([v])

    def string(self, s):
        raw = s.encode() + b"\x00"
        self.u32(len(raw))
        self.b += raw

    def pose(self, x, y, z):
        for v in (x, y, z, 0.0, 0.0, 0.0, 1.0):
            self.f64(v)


def encode_lanelet_route(*, sec, nanosec, frame, start, goal, segments, uuid=b"\x07" * 16):
    w = _CdrWriter()
    w.i32(sec)
    w.u32(nanosec)
    w.string(frame)
    w.pose(*start)
    w.pose(*goal)
    w.u32(len(segments))
    for preferred, prims in segments:
        w.i64(preferred)
        w.string("")
        w.u32(len(prims))
        for p in prims:
            w.i64(p)
            w.string("")
    for byte in uuid:
        w.u8(byte)
    w.u8(0)
    return bytes(w.b)


def test_parse_lanelet_route_roundtrip():
    raw = encode_lanelet_route(sec=1700000000, nanosec=5, frame="map",
                               start=(1.0, 2.0, 3.0), goal=(2595.0, 1764.8, 0.0),
                               segments=[(42000171, [42000171]), (11000086, [11000086, 11000084, 5])],
                               uuid=bytes(range(16)))
    m = parse_lanelet_route(raw)
    assert m.stamp_ns == 1700000000 * 10 ** 9 + 5
    assert m.frame_id == "map"
    assert m.goal_xyz == (2595.0, 1764.8, 0.0)
    assert m.preferred_ids == (42000171, 11000086)
    assert m.segment_ids == ((42000171,), (11000086, 11000084, 5))
    assert m.uuid == bytes(range(16)).hex()


def test_parse_lanelet_route_odd_frame_length_realigns():
    # A 3-char frame id leaves the cursor unaligned before the first double.
    raw = encode_lanelet_route(sec=1, nanosec=2, frame="odd", start=(0.5, 0.25, 0.0),
                               goal=(9.0, 8.0, 7.0), segments=[(1, [1])])
    m = parse_lanelet_route(raw)
    assert m.start_xyz == (0.5, 0.25, 0.0) and m.goal_xyz == (9.0, 8.0, 7.0)


# ------------------------------------------------------------ bound orientation
def _straight(y_left, y_right, reverse_left=False, reverse_right=False):
    left = np.array([[0.0, y_left, 0.0], [10.0, y_left, 0.0], [20.0, y_left, 0.0]])
    right = np.array([[0.0, y_right, 0.0], [10.0, y_right, 0.0], [20.0, y_right, 0.0]])
    lids, rids = ["l0", "l1", "l2"], ["r0", "r1", "r2"]
    if reverse_left:
        left, lids = left[::-1], lids[::-1]
    if reverse_right:
        right, rids = right[::-1], rids[::-1]
    return left, right, lids, rids


def test_orient_bounds_keeps_a_correct_lanelet():
    # Travelling +x, left bound at y=+2 is on the left: nothing should change.
    left, right, lids, rids = orient_bounds(*_straight(2.0, -2.0))
    assert lids == ["l0", "l1", "l2"] and rids == ["r0", "r1", "r2"]
    assert left[0, 0] == 0.0 and left[-1, 0] == 20.0


def test_orient_bounds_flips_a_reversed_right_bound():
    left, right, lids, rids = orient_bounds(*_straight(2.0, -2.0, reverse_right=True))
    assert rids == ["r0", "r1", "r2"], "right bound must run with the left one"
    assert np.allclose(right[:, 0], [0.0, 10.0, 20.0])


def test_orient_bounds_flips_both_when_left_is_on_the_right():
    # Stored travelling +x but the 'left' way lies at y=-2: the lanelet really
    # drives -x. Both bounds must reverse together.
    left, right, lids, rids = orient_bounds(*_straight(-2.0, 2.0))
    assert lids == ["l2", "l1", "l0"] and rids == ["r2", "r1", "r0"]
    assert left[0, 0] == 20.0 and left[-1, 0] == 0.0
    # and after the flip the left bound is on the left of travel (-x): y=-2 is left of -x travel
    travel = left[-1, :2] - left[0, :2]
    across = right[0, :2] - left[0, :2]
    assert travel[0] * across[1] - travel[1] * across[0] < 0.0


def test_centerline_is_the_midline_at_2m_spacing():
    left, right, lids, rids = _straight(2.0, -2.0)
    ll = Lanelet(1, "road", "", left, right, lids, rids)
    c = ll.centerline
    assert np.allclose(c[:, 1], 0.0) and c[0, 0] == 0.0 and c[-1, 0] == 20.0
    assert len(c) == 11  # 20 m / 2 m + 1
    assert ll.drivable and ll.start_nodes == ("l0", "r0") and ll.end_nodes == ("l2", "r2")


# ------------------------------------------------------------------- the route
def _tiny_map():
    """Two lanelets end to end (A -> B, sharing nodes) plus one off to the side."""
    def ll(i, x0, subtype="road", turn=""):
        left = np.array([[x0, 2.0, 0.0], [x0 + 20.0, 2.0, 0.0]])
        right = np.array([[x0, -2.0, 0.0], [x0 + 20.0, -2.0, 0.0]])
        lids = [f"L{x0}", f"L{x0 + 20}"]
        rids = [f"R{x0}", f"R{x0 + 20}"]
        return Lanelet(i, subtype, turn, left, right, lids, rids)
    lanelets = {1: ll(1, 0.0), 2: ll(2, 20.0, turn="left"), 3: ll(3, 100.0)}
    frame = MapFrame("map", 48.0, 11.0, "TransverseMercator")
    return LaneletXmlMap(map_dir=None, name="tiny", frame=frame, map_sha256="ab" * 32,
                         lanelets=lanelets, stop_lines=[], signals=[],
                         bounds=(-10.0, -10.0, 200.0, 10.0))


def test_route_from_preferred_lanelets():
    lmap = _tiny_map()
    raw = encode_lanelet_route(sec=10, nanosec=0, frame="map", start=(0, 0, 0), goal=(38.0, 0.0, 0.0),
                               segments=[(1, [1]), (2, [2, 3])])
    route = lmap.route(parse_lanelet_route(raw), received_ns=11 * 10 ** 9, revision=3, source="bag")
    assert route.valid and route.revision == 3
    assert [s.lane_id for s in route.lane_sequence] == ["lanelet2:1", "lanelet2:2"]
    assert route.lane_sequence[0].maneuver == Maneuver.STRAIGHT
    assert route.lane_sequence[-1].maneuver == Maneuver.DESTINATION  # last wins over turn_direction
    assert route.lane_sequence[1].transition_from_previous == TransitionType.FOLLOW
    assert np.allclose(route.destination.position_enu_m, [38.0, 0.0, 0.0])
    assert route.map_version == lmap.map_version  # else the rasterizer drops the route


def test_route_with_unknown_lanelet_is_invalid_not_gapped():
    lmap = _tiny_map()
    raw = encode_lanelet_route(sec=10, nanosec=0, frame="map", start=(0, 0, 0), goal=(0, 0, 0),
                               segments=[(1, [1]), (999, [999]), (2, [2])])
    route = lmap.route(parse_lanelet_route(raw), received_ns=0, revision=1, source="bag")
    assert not route.valid
    assert "lanelets_missing_from_map" in route.quality.failure_reasons
    assert route.quality.unresolved_discontinuities == 1


def test_no_route_placeholder_is_invalid_and_constructible():
    r = _tiny_map().no_route()
    assert not r.valid and r.lane_sequence == () and "no_route_yet" in r.quality.failure_reasons


def test_intersection_by_shared_nodes_and_turn_direction():
    lmap = _tiny_map()
    assert lmap.is_intersection(lmap.lanelets[2])       # turn_direction=left
    assert not lmap.is_intersection(lmap.lanelets[1])   # one successor, one predecessor at most


def test_navigation_map_crop_keeps_only_touching_primitives():
    lmap = _tiny_map()
    full = lmap.navigation_map()
    assert len(full.drivable_polygons) == 3 and len(full.lane_centerlines) == 3
    cropped = lmap.navigation_map(crop=(-5.0, -5.0, 45.0, 5.0))
    assert len(cropped.drivable_polygons) == 2
    assert cropped.bounds_enu_m == full.bounds_enu_m, "KNOWN_MAP_AREA must stay the full map"
    assert cropped.map_version == full.map_version


@pytest.mark.parametrize("yaw", [0.0, 0.5, -2.0, 3.0])
def test_yaw_from_quaternion(yaw):
    q = (0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2))
    assert abs(yaw_from_quaternion(*q) - yaw) < 1e-9
