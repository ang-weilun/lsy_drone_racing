#!/usr/bin/env python3
"""Render a deploy rosbag as a chase-cam video of the real flown trajectory.

The drone body is posed every frame with the logged mocap estimate (position +
orientation), so the animation is exactly what physically happened. The camera
follows the drone, so banking / pitching / heading are clearly visible. Only
logged poses are used (no dynamics), which makes it faithful for any track,
including L3 -- the gates rendered are the nominal arena, the *drone* is real.

Usage:
    python scripts/render_run.py <bag_dir> [out.mp4]

``<bag_dir>`` is the rosbag2 directory (the folder holding the ``.mcap`` +
``metadata.yaml``); a bare name is also looked up next to this script. Writes
``<bag_dir>/render.mp4`` by default. Requires a headless GL backend; the script
sets ``MUJOCO_GL=osmesa`` if it is not already configured.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

# MuJoCo/JAX read these at import time, so they must be set before the heavy
# imports below (hence the E402 suppressions).
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("SCIPY_ARRAY_API", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import gymnasium  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import jax.numpy as jp  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from rosbags.highlevel import AnyReader  # noqa: E402
from scipy.spatial.transform import Rotation, Slerp  # noqa: E402

from lsy_drone_racing.utils import draw_line, load_config  # noqa: E402

HERE = Path(__file__).resolve().parent
CONFIG = REPO / "config" / "level0.toml"  # nominal arena (gates/floor) for context

FPS = 60
CAM_DISTANCE = 1.7  # m, chase distance
CAM_AZIMUTH = 120.0  # deg, fixed world heading (3/4 view)
CAM_ELEVATION = -22.0  # deg
LOOKAT_SMOOTH = 0.15  # EMA factor on the camera target (lower = smoother)
RED = np.array([0.95, 0.15, 0.15, 1.0])
TRAIL_SIZE = 5.0
MAX_TRAIL_PTS = 200  # keep under sim.max_visual_geom


def resolve_bag(arg: str) -> Path:
    """Resolve a bag path or a bare bag name (looked up next to this script)."""
    p = Path(arg)
    if p.is_dir():
        return p
    alt = HERE / arg
    if alt.is_dir():
        return alt
    raise SystemExit(f"bag dir not found: {arg!r} (tried {p} and {alt})")


def read_bag(bag: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(pose_t, pose_v[x,y,z,qx,qy,qz,qw], cmd_t)`` from the rosbag."""
    pose_t: list[float] = []
    pose_v: list[list[float]] = []
    cmd_t: list[float] = []
    with AnyReader([bag]) as reader:
        t0 = reader.start_time
        for conn, ts, raw in reader.messages():
            mt = conn.msgtype
            t = (ts - t0) * 1e-9
            if conn.topic.endswith("/estimate/pose") and mt == "geometry_msgs/msg/PoseStamped":
                m = reader.deserialize(raw, mt)
                p, q = m.pose.position, m.pose.orientation
                pose_t.append(t)
                pose_v.append([p.x, p.y, p.z, q.x, q.y, q.z, q.w])
            elif conn.topic.endswith("/command") and mt == "std_msgs/msg/Float64MultiArray":
                cmd_t.append(t)
    if not pose_t:
        raise SystemExit("no */estimate/pose messages found in bag")
    return np.asarray(pose_t), np.asarray(pose_v), np.asarray(cmd_t)


def resample_pose(
    pt: np.ndarray, pv: np.ndarray, t_query: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate logged pose onto query times (pos linear, quat slerp)."""
    pos = np.column_stack([np.interp(t_query, pt, pv[:, j]) for j in range(3)])
    slerp = Slerp(pt, Rotation.from_quat(pv[:, 3:7]))
    quat = slerp(np.clip(t_query, pt[0], pt[-1])).as_quat()
    return pos, quat


def trail(hist: list[np.ndarray], eps: float = 1e-3) -> np.ndarray:
    """Stride the path history to a bounded point count, dropping zero segments."""
    pts = np.asarray(hist)
    if len(pts) > MAX_TRAIL_PTS:
        pts = pts[:: int(np.ceil(len(pts) / MAX_TRAIL_PTS))]
    if len(pts) < 2:
        return pts
    keep = np.concatenate(([True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > eps))
    return pts[keep]


def annotate(frame: np.ndarray, t: float, name: str) -> np.ndarray:
    """Overlay the bag name, timestamp, and a trail-colour legend on a frame."""
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    d.text((12, 10), f"{name}   t = {t:5.2f} s", fill=(255, 255, 255))
    d.text((12, 26), "RED = flown path", fill=(235, 70, 70))
    return np.asarray(img)


def main() -> None:
    """Render the bag named on the command line to a chase-cam mp4."""
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    bag = resolve_bag(sys.argv[1])
    out = str(Path(sys.argv[2]) if len(sys.argv) > 2 else bag / "render.mp4")
    width, height = 1280, 720

    pose_t, pose_v, cmd_t = read_bag(bag)
    # render the controlled window if commands exist, else the whole pose span
    if cmd_t.size:
        lo, hi = cmd_t[0], cmd_t[-1]
    else:
        lo, hi = pose_t[0], pose_t[-1]
    lo = max(lo, pose_t[0])
    hi = min(hi, pose_t[-1])
    grid = np.arange(lo, hi, 1.0 / FPS)
    real_pos, real_quat = resample_pose(pose_t, pose_v, grid)

    cfg = load_config(CONFIG)
    cfg.sim.render = False
    env = gymnasium.make(
        "DroneRacing-v0",
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        sensor_range=cfg.env.sensor_range,
        control_mode="attitude",
        track=copy.deepcopy(cfg.env.track),
        disturbances=cfg.env.get("disturbances"),
        randomizations=cfg.env.get("randomizations"),
        seed=cfg.env.seed,
    )
    env.reset()
    sim = env.unwrapped.sim

    writer = imageio.get_writer(out, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    look = real_pos[0].copy()
    init_cam = {
        "distance": CAM_DISTANCE,
        "azimuth": CAM_AZIMUTH,
        "elevation": CAM_ELEVATION,
        "lookat": list(look),
    }
    hist: list[np.ndarray] = []

    for k in range(len(grid)):
        # overwrite the rendered drone pose with the logged real state and mark
        # the mjx buffer stale so render() re-syncs it (otherwise it freezes).
        st = sim.data.states
        st = st.replace(
            pos=st.pos.at[0, 0, :].set(jp.asarray(real_pos[k])),
            quat=st.quat.at[0, 0, :].set(jp.asarray(real_quat[k])),
        )
        sim.data = sim.data.replace(states=st, core=sim.data.core.replace(mjx_synced=False))
        hist.append(real_pos[k].copy())

        # chase camera: EMA-smoothed lookat on the drone
        look = (1 - LOOKAT_SMOOTH) * look + LOOKAT_SMOOTH * real_pos[k]
        if sim.viewer is not None:
            cam = sim.viewer.viewer.cam
            cam.lookat[:] = look
            cam.distance = CAM_DISTANCE
            cam.azimuth = CAM_AZIMUTH
            cam.elevation = CAM_ELEVATION

        rt = trail(hist)
        if len(rt) > 1:
            draw_line(env, rt, rgba=RED, min_size=TRAIL_SIZE, max_size=TRAIL_SIZE)

        frame = np.asarray(
            sim.render(mode="rgb_array", camera=-1, cam_config=init_cam, width=width, height=height)
        )
        writer.append_data(annotate(frame, float(grid[k]), bag.name))

    writer.close()
    env.close()
    print(f"wrote {out}  ({len(grid)} frames, {len(grid) / FPS:.1f}s @ {FPS}fps)")


if __name__ == "__main__":
    main()
