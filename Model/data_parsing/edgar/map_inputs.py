"""EDGAR map + route inputs: lanelet2 XML -> NavigationMap, LaneletRoute -> NavigationRoute.

The KITScenes navigation path needs the ``lanelet2`` Python bindings, which are
installed in neither the ROS env nor the torch venv. This module reads the
lanelet2 ``.osm`` XML directly, projects it through the map's own
``map_projector_info.yaml`` (the bags' ``map`` frame -- verified against the
lanelet IDs Autoware drew in ``/map/vector_map_marker`` to <2 m), and turns
Autoware's planned ``LaneletRoute`` into the navigation contract's route. The
rasterizer, supervision and artifact encoders are the repo's own.

    from data_parsing.edgar.map_inputs import LaneletXmlMap, parse_lanelet_route
    lmap = LaneletXmlMap.load("/.../autoware.maps/lanelets/munich_map")
    nav_map = lmap.navigation_map(crop=(xmin, ymin, xmax, ymax))
    route = lmap.route(parse_lanelet_route(raw_cdr), stamp_ns, revision=1, source="bag")

ponytail: centerlines are the mean of the arc-length-resampled bounds, not
lanelet2's exact centerline; intersections come from turn_direction plus shared
end nodes, not the routing graph. Both are what the 1 m/px raster can resolve.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

from navigation.contracts import (
    Destination,
    DirectedLaneField,
    Maneuver,
    MapFrame,
    NavigationMap,
    NavigationRoute,
    PolygonPrimitive,
    PolylinePrimitive,
    RouteLaneSegment,
    RouteProvenance,
    RouteQuality,
    StaticTrafficSignal,
    TransitionType,
)

MAP_INPUTS_VERSION = "edgar_map_inputs_v1"
CENTERLINE_SPACING_M = 2.0
# Lanelet subtypes a car may drive on. bicycle_lane/crosswalk/etc. still draw
# their boundaries but are neither drivable nor direction fields.
DRIVABLE_SUBTYPES = {"road", "road_shoulder", "highway", "play_street",
                     "emergency_lane", "bus_lane", "exit"}


# ------------------------------------------------------------ CDR: LaneletRoute
class _Cdr:
    """Just enough XCDR1 little-endian reading for LaneletRoute."""

    def __init__(self, raw: bytes):
        self.b, self.o = raw, 4  # 4-byte encapsulation header

    def _align(self, n: int) -> None:
        self.o += (-(self.o - 4)) % n  # alignment counts from the payload start

    def _unpack(self, fmt: str, size: int):
        self._align(size)
        value, = struct.unpack_from(fmt, self.b, self.o)
        self.o += size
        return value

    def u32(self) -> int: return self._unpack("<I", 4)
    def i32(self) -> int: return self._unpack("<i", 4)
    def i64(self) -> int: return self._unpack("<q", 8)
    def f64(self) -> float: return self._unpack("<d", 8)

    def u8(self) -> int:
        value = self.b[self.o]
        self.o += 1
        return value

    def string(self) -> str:
        n = self.u32()
        value = self.b[self.o:self.o + max(0, n - 1)].decode("utf-8", "replace")
        self.o += n
        return value

    def pose_xyz(self) -> tuple[float, float, float]:
        xyz = tuple(self.f64() for _ in range(3))
        for _ in range(4):  # quaternion
            self.f64()
        return xyz


@dataclasses.dataclass(frozen=True)
class LaneletRouteMsg:
    stamp_ns: int
    frame_id: str
    start_xyz: tuple[float, float, float]
    goal_xyz: tuple[float, float, float]
    preferred_ids: tuple[int, ...]          # one lanelet per segment, the planner's pick
    segment_ids: tuple[tuple[int, ...], ...]  # every lanelet the segment may use
    uuid: str


def parse_lanelet_route(raw: bytes) -> LaneletRouteMsg:
    """autoware_planning_msgs/LaneletRoute from CDR.

    Header{Time, string frame_id}, Pose start, Pose goal, LaneletSegment[]
    {LaneletPrimitive preferred{int64 id, string type}, LaneletPrimitive[]},
    UUID uint8[16], bool allow_modification. No package resolves this type
    outside alpamayo's container, hence by hand; validated against the
    ``/planning/mission_planning/goal`` topic on real bags.
    """
    c = _Cdr(raw)
    sec, nanosec = c.i32(), c.u32()
    frame_id = c.string()
    start, goal = c.pose_xyz(), c.pose_xyz()
    preferred, segments = [], []
    for _ in range(c.u32()):
        pref_id = c.i64()
        c.string()
        prims = []
        for _ in range(c.u32()):
            prims.append(c.i64())
            c.string()
        preferred.append(pref_id)
        segments.append(tuple(prims))
    uuid = bytes(c.u8() for _ in range(16)).hex()
    return LaneletRouteMsg(sec * 10 ** 9 + nanosec, frame_id, start, goal,
                           tuple(preferred), tuple(segments), uuid)


# ------------------------------------------------------------------- geometry
def _resample(points: np.ndarray, count: int) -> np.ndarray:
    """``count`` points spaced evenly along the polyline's arc length."""
    if len(points) == 1:
        return np.repeat(points, count, axis=0)
    seg = np.hypot(*np.diff(points[:, :2], axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0.0:
        return np.repeat(points[:1], count, axis=0)
    target = np.linspace(0.0, s[-1], count)
    return np.stack([np.interp(target, s, points[:, i]) for i in range(points.shape[1])], axis=1)


def _polyline_length(points: np.ndarray) -> float:
    return float(np.hypot(*np.diff(points[:, :2], axis=0).T).sum()) if len(points) > 1 else 0.0


def orient_bounds(left: np.ndarray, right: np.ndarray,
                  left_ids: list[str], right_ids: list[str]):
    """Return (left, right, left_ids, right_ids) both running in the driving direction.

    lanelet2 stores each bound as an independent way with arbitrary direction. It
    fixes that on load: first the right bound is flipped to run the same way as
    the left, then the pair is flipped together so 'left' really lies on the
    left of travel. Same rule here, from geometry alone.
    """
    same = np.hypot(*(left[0, :2] - right[0, :2])) + np.hypot(*(left[-1, :2] - right[-1, :2]))
    flipped = np.hypot(*(left[0, :2] - right[-1, :2])) + np.hypot(*(left[-1, :2] - right[0, :2]))
    if flipped < same:
        right, right_ids = right[::-1], right_ids[::-1]
    travel = left[-1, :2] - left[0, :2]
    if np.hypot(*travel) < 1e-6:
        travel = right[-1, :2] - right[0, :2]
    across = right[0, :2] - left[0, :2]
    # In a right-handed ENU frame the right-hand side of travel is a negative
    # z-cross. Positive means the 'left' bound is on the right: reverse both.
    if travel[0] * across[1] - travel[1] * across[0] > 0.0:
        left, right = left[::-1], right[::-1]
        left_ids, right_ids = left_ids[::-1], right_ids[::-1]
    return left, right, left_ids, right_ids


@dataclasses.dataclass
class Lanelet:
    lanelet_id: int
    subtype: str
    turn_direction: str
    left: np.ndarray            # (N,3) in the driving direction
    right: np.ndarray           # (M,3) in the driving direction
    left_ids: list[str]         # node ids, same order -- connectivity by shared nodes
    right_ids: list[str]

    @property
    def drivable(self) -> bool:
        return self.subtype in DRIVABLE_SUBTYPES

    @property
    def polygon(self) -> np.ndarray:
        return np.concatenate([self.left, self.right[::-1]], axis=0)

    @property
    def centerline(self) -> np.ndarray:
        length = max(_polyline_length(self.left), _polyline_length(self.right))
        count = max(2, int(math.ceil(length / CENTERLINE_SPACING_M)) + 1)
        return 0.5 * (_resample(self.left, count) + _resample(self.right, count))

    @property
    def start_nodes(self) -> tuple[str, str]:
        return self.left_ids[0], self.right_ids[0]

    @property
    def end_nodes(self) -> tuple[str, str]:
        return self.left_ids[-1], self.right_ids[-1]


# ---------------------------------------------------------------------- the map
class LaneletXmlMap:
    """A lanelet2 ``.osm`` + ``map_projector_info.yaml`` directory, in metres."""

    def __init__(self, map_dir: Path, name: str, frame: MapFrame, map_sha256: str,
                 lanelets: dict[int, Lanelet], stop_lines: list[np.ndarray],
                 signals: list[tuple[int, np.ndarray]], bounds, projector=None):
        self.map_dir = map_dir
        self.projector = projector  # pyproj Proj; None for synthetic maps in tests
        self.name = name
        self.frame = frame
        self.map_sha256 = map_sha256
        self.map_version = f"{name}:{map_sha256[:16]}"
        self.lanelets = lanelets
        self.stop_lines = stop_lines
        self.signals = signals
        self.bounds = bounds
        # successor/predecessor counts by shared end nodes, for intersections
        starts: dict[tuple[str, str], int] = {}
        ends: dict[tuple[str, str], int] = {}
        for ll in lanelets.values():
            starts[ll.start_nodes] = starts.get(ll.start_nodes, 0) + 1
            ends[ll.end_nodes] = ends.get(ll.end_nodes, 0) + 1
        self._successors = {i: starts.get(ll.end_nodes, 0) for i, ll in lanelets.items()}
        self._predecessors = {i: ends.get(ll.start_nodes, 0) for i, ll in lanelets.items()}

    @classmethod
    def load(cls, map_dir: str | Path) -> "LaneletXmlMap":
        map_dir = Path(map_dir)
        osm = map_dir / "lanelet2_map.osm"
        cfg = yaml.safe_load((map_dir / "map_projector_info.yaml").read_text())
        origin = cfg["map_origin"]
        projector = str(cfg.get("projector_type", ""))
        if projector != "TransverseMercator":
            # MGRS maps need a different projector; not needed for Munich/Berlin.
            raise ValueError(f"{map_dir.name}: projector_type {projector!r} is not supported")
        from pyproj import Proj
        tm = Proj(proj="tmerc", lat_0=origin["latitude"], lon_0=origin["longitude"],
                  k=float(cfg.get("scale_factor", 1.0)), x_0=0, y_0=0, ellps="WGS84")
        frame = MapFrame(frame_id="map", origin_latitude_deg=float(origin["latitude"]),
                         origin_longitude_deg=float(origin["longitude"]),
                         projection="TransverseMercator")

        raw = osm.read_bytes()
        root = ET.fromstring(raw)
        nodes: dict[str, np.ndarray] = {}
        for n in root.findall("node"):
            if n.get("lat") is None:
                continue
            x, y = tm(float(n.get("lon")), float(n.get("lat")))
            ele = next((float(t.get("v")) for t in n.findall("tag") if t.get("k") == "ele"), 0.0)
            nodes[n.get("id")] = np.array([x, y, ele])
        ways: dict[str, tuple[list[str], dict[str, str]]] = {}
        for w in root.findall("way"):
            refs = [nd.get("ref") for nd in w.findall("nd") if nd.get("ref") in nodes]
            tags = {t.get("k"): t.get("v") for t in w.findall("tag")}
            ways[w.get("id")] = (refs, tags)

        def way_points(way_id: str) -> tuple[np.ndarray, list[str]]:
            refs, _ = ways.get(way_id, ([], {}))
            return np.array([nodes[r] for r in refs]).reshape(-1, 3), refs

        lanelets: dict[int, Lanelet] = {}
        signals: list[tuple[int, np.ndarray]] = []
        for r in root.findall("relation"):
            tags = {t.get("k"): t.get("v") for t in r.findall("tag")}
            members = [(m.get("type"), m.get("ref"), m.get("role")) for m in r.findall("member")]
            if tags.get("type") == "lanelet":
                left_id = next((ref for kind, ref, role in members if kind == "way" and role == "left"), None)
                right_id = next((ref for kind, ref, role in members if kind == "way" and role == "right"), None)
                if left_id is None or right_id is None:
                    continue
                left, lids = way_points(left_id)
                right, rids = way_points(right_id)
                if len(left) < 2 or len(right) < 2:
                    continue
                left, right, lids, rids = orient_bounds(left, right, lids, rids)
                lanelets[int(r.get("id"))] = Lanelet(
                    int(r.get("id")), tags.get("subtype", ""), tags.get("turn_direction", ""),
                    left, right, lids, rids)
            elif tags.get("type") == "regulatory_element" and "traffic_light" in tags.get("subtype", ""):
                pts = [way_points(ref)[0] for kind, ref, role in members
                       if kind == "way" and role == "refers"
                       and ways.get(ref, ([], {}))[1].get("type") == "traffic_light"]
                pts = [p for p in pts if len(p)]
                if pts:
                    signals.append((int(r.get("id")), np.concatenate(pts).mean(axis=0)))
        stop_lines = [way_points(wid)[0] for wid, (_, tags) in ways.items()
                      if tags.get("type") == "stop_line"]
        stop_lines = [p for p in stop_lines if len(p) >= 2]
        xy = np.array([p[:2] for p in nodes.values()])
        bounds = (float(xy[:, 0].min()), float(xy[:, 1].min()),
                  float(xy[:, 0].max()), float(xy[:, 1].max()))
        return cls(map_dir, map_dir.name, frame, hashlib.sha256(raw).hexdigest(),
                   lanelets, stop_lines, signals, bounds, projector=tm)

    def is_intersection(self, ll: Lanelet) -> bool:
        return (ll.turn_direction in ("left", "right", "u_turn")
                or self._successors.get(ll.lanelet_id, 0) > 1
                or self._predecessors.get(ll.lanelet_id, 0) > 1)

    def navigation_map(self, crop: tuple[float, float, float, float] | None = None) -> NavigationMap:
        """The canonical map, optionally only primitives touching ``crop`` (xmin,ymin,xmax,ymax).

        Cropping is a speed knob: the rasterizer re-packs every primitive per
        render, and a bag drives through a few km of a 14 km map. ``bounds_enu_m``
        stays the full map so KNOWN_MAP_AREA is still honest.
        """
        def touches(points: np.ndarray) -> bool:
            if crop is None:
                return True
            xmin, ymin, xmax, ymax = crop
            return bool((points[:, 0] >= xmin).any() and (points[:, 0] <= xmax).any()
                        and (points[:, 1] >= ymin).any() and (points[:, 1] <= ymax).any())

        drivable, intersections, crosswalks, centerlines, directions = [], [], [], [], []
        boundaries: dict[str, PolylinePrimitive] = {}
        for lid in sorted(self.lanelets):
            ll = self.lanelets[lid]
            if not touches(ll.polygon):
                continue
            cid = f"lanelet2:{lid}"
            polygon = PolygonPrimitive(f"{cid}:polygon", ll.polygon)
            if ll.subtype == "crosswalk":
                crosswalks.append(polygon)
            elif ll.drivable:
                drivable.append(polygon)
                if self.is_intersection(ll):
                    intersections.append(polygon)
            for side, pts, ids in (("left", ll.left, ll.left_ids), ("right", ll.right, ll.right_ids)):
                key = "-".join(ids) if ids else f"{lid}:{side}"
                key = hashlib.sha1(key.encode()).hexdigest()[:16]
                if key not in boundaries:
                    boundaries[key] = PolylinePrimitive(f"lanelet2:boundary:{key}", pts)
            if ll.subtype != "crosswalk":
                center = ll.centerline
                centerlines.append(PolylinePrimitive(f"{cid}:centerline", center))
                if ll.drivable:
                    directions.append(DirectedLaneField(cid, center))
        stop_lines = tuple(PolylinePrimitive(f"lanelet2:stop_line:{i}", pts)
                           for i, pts in enumerate(self.stop_lines) if touches(pts))
        signals = tuple(StaticTrafficSignal(f"lanelet2:signal:{sid}", pos)
                        for sid, pos in self.signals if touches(pos.reshape(1, -1)))
        return NavigationMap(
            map_version=self.map_version, provider="edgar_lanelet2_xml", frame=self.frame,
            bounds_enu_m=self.bounds,
            drivable_polygons=tuple(drivable), lane_boundaries=tuple(boundaries.values()),
            lane_centerlines=tuple(centerlines), intersection_polygons=tuple(intersections),
            crosswalk_polygons=tuple(crosswalks), stop_lines=stop_lines,
            static_traffic_signals=signals, directed_lane_fields=tuple(directions),
            layer_availability={"drivable_area": True, "lane_boundary": True,
                                "lane_centerline": True, "intersection": True,
                                "crosswalk": True, "stop_line": bool(stop_lines),
                                "static_traffic_signal": bool(signals),
                                "traffic_direction": True},
            provenance={"adapter": MAP_INPUTS_VERSION, "map_dir": str(self.map_dir),
                        "map_sha256": self.map_sha256, "lanelets": len(self.lanelets),
                        "cropped": crop is not None})

    def route(self, msg: LaneletRouteMsg, received_ns: int, *, revision: int,
              source: str) -> NavigationRoute:
        """Autoware's planned route as the navigation contract's route.

        The planner's ``preferred_primitive`` per segment is the lane it meant
        to drive; the other primitives are lane-change alternatives and are not
        drawn. Lanelets missing from this map file (the vehicle drove on a
        slightly different revision) invalidate the route rather than leaving a
        gap the corridor would silently bridge.
        """
        missing = [i for i in msg.preferred_ids if i not in self.lanelets]
        segments = []
        for index, lid in enumerate(msg.preferred_ids):
            ll = self.lanelets.get(lid)
            if ll is None:
                continue
            last = index == len(msg.preferred_ids) - 1
            maneuver = (Maneuver.DESTINATION if last
                        else {"left": Maneuver.LEFT, "right": Maneuver.RIGHT,
                              "u_turn": Maneuver.U_TURN}.get(ll.turn_direction, Maneuver.STRAIGHT))
            transition = TransitionType.FOLLOW
            previous = self.lanelets.get(msg.preferred_ids[index - 1]) if index else None
            if segments and previous is not None and previous.end_nodes != ll.start_nodes:
                # Not a successor by shared nodes: the planner changed lane.
                prev = previous.centerline[-1, :2]
                d = ll.centerline[-1, :2] - ll.centerline[0, :2]
                across = prev - ll.centerline[0, :2]
                transition = (TransitionType.LEFT_ADJACENT
                              if d[0] * across[1] - d[1] * across[0] < 0.0
                              else TransitionType.RIGHT_ADJACENT)
            segments.append(RouteLaneSegment(
                lane_id=f"lanelet2:{lid}", provider_segment_id=str(lid),
                centerline_enu_m=ll.centerline, left_boundary_enu_m=ll.left,
                right_boundary_enu_m=ll.right, transition_from_previous=transition,
                maneuver=maneuver))
        failure = []
        if not segments:
            failure.append("no_lane_sequence")
        if missing:
            failure.append("lanelets_missing_from_map")
        valid = not failure
        quality = RouteQuality(
            # The route is the planner's own output, not a trace match: it is
            # exact by construction, so the match statistics are degenerate.
            matched_pose_ratio=1.0 if valid else 0.0,
            median_lateral_distance_m=0.0, p95_lateral_distance_m=0.0,
            median_heading_error_rad=0.0, p95_heading_error_rad=0.0,
            unresolved_discontinuities=len(missing), failure_reasons=tuple(failure))
        return NavigationRoute(
            route_id=f"autoware:{msg.uuid or hashlib.sha1(str(msg.stamp_ns).encode()).hexdigest()[:16]}",
            revision=revision, provider="autoware_mission_planning",
            timestamp_ns=max(0, msg.stamp_ns), valid_from_ns=max(0, received_ns),
            map_version=self.map_version, frame=self.frame, lane_sequence=tuple(segments),
            destination=Destination(np.asarray(msg.goal_xyz), "autoware_mission_planning_goal"),
            confidence=1.0 if valid else 0.0, valid=valid, quality=quality,
            estimated_destination=False,
            provenance=RouteProvenance(
                source_revision=source, matcher_version=MAP_INPUTS_VERSION,
                matcher_config_sha256=hashlib.sha256(MAP_INPUTS_VERSION.encode()).hexdigest(),
                map_sha256=self.map_sha256,
                trace_sha256=hashlib.sha256(f"{msg.uuid}:{msg.stamp_ns}".encode()).hexdigest()))


    def no_route(self) -> NavigationRoute:
        """An explicitly invalid route for anchors before the planner spoke.

        The rasterizer and supervision both take a route object and read
        ``.valid``; passing this instead of None keeps every sample on the same
        code path and records *why* the route channels are empty.
        """
        digest = hashlib.sha256(b"no_route").hexdigest()
        return NavigationRoute(
            route_id="none", revision=0, provider="autoware_mission_planning",
            timestamp_ns=0, valid_from_ns=0, map_version=self.map_version, frame=self.frame,
            lane_sequence=(), destination=Destination(np.zeros(3), "none"),
            confidence=0.0, valid=False,
            quality=RouteQuality(0.0, 0.0, 0.0, 0.0, 0.0, failure_reasons=("no_route_yet",)),
            estimated_destination=True,
            provenance=RouteProvenance(source_revision="none", matcher_version=MAP_INPUTS_VERSION,
                                       matcher_config_sha256=digest, map_sha256=self.map_sha256,
                                       trace_sha256=digest))


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Heading about +z, CCW from +x (east), as the rasterizer's EgoPose expects."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ------------------------------------------------------------ per-bag renderer
class EdgarNavigation:
    """One loaded map + rasterizer, shared across bags of the same city."""

    def __init__(self, map_dir: str | Path):
        from navigation.rasterizer import NativeNavigationRasterizer
        self.lmap = LaneletXmlMap.load(map_dir)
        self.rasterizer = NativeNavigationRasterizer()

    @property
    def geometry_id(self) -> str:
        return self.rasterizer.geometry.geometry_id

    def utm_to_map(self, pose_xyyaw: np.ndarray, utm_zone: int) -> np.ndarray:
        """[x, y, yaw] in a UTM zone -> the map's TransverseMercator frame.

        For bags without Autoware localization (the human-driven Berlin resim
        bag) the only pose is the INS odometry in UTM. Positions go through
        lat/lon; yaw is corrected by the difference in grid convergence between
        the two projections -- about 1.3 deg in Berlin (UTM 33 is centred on
        15 E, the map on 13.4 E), i.e. ~5 m of lateral error at the raster's
        far edge if ignored.
        """
        from pyproj import Proj
        utm = Proj(proj="utm", zone=int(utm_zone), ellps="WGS84")
        p = np.asarray(pose_xyyaw, dtype=np.float64).reshape(-1, 3)
        lon, lat = utm(p[:, 0], p[:, 1], inverse=True)
        x, y = self.lmap.projector(lon, lat)
        conv_utm = np.radians(utm.get_factors(lon, lat).meridian_convergence)
        conv_map = np.radians(self.lmap.projector.get_factors(lon, lat).meridian_convergence)
        yaw = p[:, 2] + (conv_utm - conv_map)
        yaw = (yaw + np.pi) % (2.0 * np.pi) - np.pi
        return np.stack([x, y, yaw], axis=1)

    def for_bag(self, bag_name: str, routes_raw: list[tuple[int, bytes]],
                trace_xy: np.ndarray, margin_m: float = 300.0) -> "BagNavigation":
        return BagNavigation(self, bag_name, routes_raw, trace_xy, margin_m)


class BagNavigation:
    """The map cropped to one bag's drive, and that bag's planner routes by time."""

    def __init__(self, nav: EdgarNavigation, bag_name: str,
                 routes_raw: list[tuple[int, bytes]], trace_xy: np.ndarray, margin_m: float):
        import bisect
        self.nav = nav
        self._bisect = bisect
        xy = np.asarray(trace_xy, dtype=np.float64).reshape(-1, 2)
        crop = (xy[:, 0].min() - margin_m, xy[:, 1].min() - margin_m,
                xy[:, 0].max() + margin_m, xy[:, 1].max() + margin_m)
        self.nav_map = nav.lmap.navigation_map(crop)
        self.routes: list[tuple[int, NavigationRoute]] = []
        for revision, (received_ns, raw) in enumerate(sorted(routes_raw), start=1):
            msg = parse_lanelet_route(raw)
            self.routes.append((received_ns, nav.lmap.route(msg, received_ns, revision=revision,
                                                             source=bag_name)))
        self._received = [r for r, _ in self.routes]
        self._none = nav.lmap.no_route()
        self.stats = {"routes": len(self.routes),
                      "routes_valid": sum(r.valid for _, r in self.routes),
                      "map_primitives": {
                          "drivable": len(self.nav_map.drivable_polygons),
                          "boundaries": len(self.nav_map.lane_boundaries),
                          "centerlines": len(self.nav_map.lane_centerlines),
                          "intersections": len(self.nav_map.intersection_polygons),
                          "stop_lines": len(self.nav_map.stop_lines),
                          "signals": len(self.nav_map.static_traffic_signals)}}

    def route_at(self, t_ns: int) -> NavigationRoute:
        """The latest planner route received at or before ``t_ns``."""
        i = self._bisect.bisect_right(self._received, t_ns) - 1
        return self.routes[i][1] if i >= 0 else self._none

    def _near(self, route: NavigationRoute, pose) -> NavigationRoute:
        """``route`` with only the lane segments that can reach the raster.

        build_route_supervision walks every centerline segment of the route and
        paints the ones inside the raster; a Munich route is 200 lanelets and
        kilometres long, so 75% of the render time went to samples that were
        discarded. Segments with no point within the raster's reach cannot paint
        anything, so dropping them changes no output. The raster itself was
        already rendered from the full route.
        """
        if not route.valid:
            return route
        g = self.nav.rasterizer.geometry
        reach = math.hypot(max(abs(g.x_min_m), abs(g.x_max_m)),
                           max(abs(g.y_min_m), abs(g.y_max_m))) + 2.0 * CENTERLINE_SPACING_M
        ego = np.array([pose.x_enu_m, pose.y_enu_m])
        kept = tuple(s for s in route.lane_sequence
                     if (np.hypot(*(s.centerline_enu_m[:, :2] - ego).T) <= reach).any())
        if len(kept) == len(route.lane_sequence):
            return route
        if not kept:  # off the route entirely: keep the nearest so the route stays valid
            kept = (min(route.lane_sequence,
                        key=lambda s: np.hypot(*(s.centerline_enu_m[:, :2] - ego).T).min()),)
        return dataclasses.replace(route, lane_sequence=kept)

    def members(self, pose) -> dict[str, bytes]:
        """The schema-v8 navigation members for one anchor pose (an EgoPose)."""
        from navigation.artifacts import encode_sample_navigation
        from navigation.supervision import build_route_supervision
        route = self.route_at(pose.timestamp_ns)
        raster = self.nav.rasterizer.render(self.nav_map, route, pose)
        supervision = build_route_supervision(self._near(route, pose), pose, raster,
                                              self.nav.rasterizer.geometry)
        return encode_sample_navigation(
            raster, route_supervision=supervision,
            extra_metadata={"route_source": route.provider,
                            "route_received_ns": int(route.valid_from_ns),
                            "map_name": self.nav.lmap.name})
