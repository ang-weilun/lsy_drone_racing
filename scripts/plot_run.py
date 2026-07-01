#!/usr/bin/env python3
"""Plot attitude commands vs. achieved attitude (and thrust) from a deploy rosbag.

Reads the ``*/command`` (roll, pitch, yaw [deg], thrust [pwm]) and
``*/estimate/pose`` topics straight from the rosbag and overlays commanded vs.
achieved roll/pitch/yaw plus the thrust command. Drone-id agnostic (matches any
topic ending in ``/command`` / ``/estimate/pose``), so it works for L0-L3.

Usage:
    python scripts/plot_run.py <bag_dir> [out.png]

``<bag_dir>`` is the rosbag2 directory (the folder holding the ``.mcap`` +
``metadata.yaml``); a bare name is also looked up next to this script. Writes
``<bag_dir>/attitude_commands.png`` by default.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent


def resolve_bag(arg: str) -> Path:
    """Resolve a bag path or a bare bag name (looked up next to this script)."""
    p = Path(arg)
    if p.is_dir():
        return p
    alt = HERE / arg
    if alt.is_dir():
        return alt
    raise SystemExit(f"bag dir not found: {arg!r} (tried {p} and {alt})")


def read_bag(bag: Path) -> dict[str, np.ndarray]:
    """Extract command and pose-estimate time series from the rosbag."""
    cmd_t: list[float] = []
    cmd_v: list[list[float]] = []
    pose_t: list[float] = []
    pose_v: list[list[float]] = []
    with AnyReader([bag]) as reader:
        t0 = reader.start_time
        for conn, ts, raw in reader.messages():
            mt = conn.msgtype
            t = (ts - t0) * 1e-9
            if conn.topic.endswith("/command") and mt == "std_msgs/msg/Float64MultiArray":
                data = list(reader.deserialize(raw, mt).data)
                if len(data) >= 4:
                    cmd_t.append(t)
                    cmd_v.append(data[:4])
            elif conn.topic.endswith("/estimate/pose") and mt == "geometry_msgs/msg/PoseStamped":
                m = reader.deserialize(raw, mt)
                p, q = m.pose.position, m.pose.orientation
                pose_t.append(t)
                pose_v.append([p.x, p.y, p.z, q.x, q.y, q.z, q.w])
    if not pose_t:
        raise SystemExit("no */estimate/pose messages found in bag")
    return {
        "cmd_t": np.asarray(cmd_t),
        "cmd_v": np.asarray(cmd_v).reshape(-1, 4),
        "pose_t": np.asarray(pose_t),
        "pose_v": np.asarray(pose_v),
    }


def main() -> None:
    """Read the bag named on the command line and write the attitude plot."""
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    bag = resolve_bag(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else bag / "attitude_commands.png"

    d = read_bag(bag)
    cmd_t, cmd_v, pose_t, pose_v = d["cmd_t"], d["cmd_v"], d["pose_t"], d["pose_v"]

    # achieved attitude from the pose-estimate quaternion (xyzw) -> euler deg
    ach_eul = Rotation.from_quat(pose_v[:, 3:7]).as_euler("xyz", degrees=True)
    ach_eul = np.degrees(np.unwrap(np.radians(ach_eul), axis=0))

    # focus the time axis on the controlled window (when commands were sent)
    if cmd_t.size:
        lo, hi = cmd_t[0] - 0.5, cmd_t[-1] + 0.5
        t_ref = cmd_t[0]
    else:
        lo, hi, t_ref = pose_t[0], pose_t[-1], pose_t[0]
    sel = (pose_t >= lo) & (pose_t <= hi)

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    labels = ["roll", "pitch", "yaw"]
    for i, (ax, lab) in enumerate(zip(axes[:3], labels)):
        ax.plot(pose_t[sel] - t_ref, ach_eul[sel, i], color="C0", lw=1.2, label="achieved")
        if cmd_t.size:
            ax.plot(cmd_t - t_ref, cmd_v[:, i], color="C3", lw=1.0, ls="--", label=f"{lab} cmd")
        ax.set_ylabel(f"{lab} [deg]")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    if cmd_t.size:
        axes[3].plot(cmd_t - t_ref, cmd_v[:, 3], color="C2", lw=1.0)
    axes[3].set_ylabel("thrust cmd [pwm]")
    axes[3].set_xlabel(f"time [s]  (t0 = {t_ref:.2f} s)")
    axes[3].grid(alpha=0.3)

    fig.suptitle(f"{bag.name}: attitude commands vs. achieved", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    window = f", window=[{cmd_t[0]:.2f},{cmd_t[-1]:.2f}]s" if cmd_t.size else " (none)"
    print(f"  commands: n={cmd_t.size}{window}")
    print(f"  pose: n={pose_t.size}, span=[{pose_t[0]:.2f},{pose_t[-1]:.2f}]s")


if __name__ == "__main__":
    main()
