import json
import yaml
import math
import numpy as np
from pathlib import Path


def matrix_to_quat_xyzw(R):
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)

    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    q = np.array([qx, qy, qz, qw], dtype=float)
    q = q / np.linalg.norm(q)
    return q.tolist()


def charuco_corner_position(point_id, squares_x, square_length):
    """
    ChArUco internal corner IDs for 7x5 board:

    0   1   2   3   4   5
    6   7   8   9   10  11
    12  13  14  15  16  17
    18  19  20  21  22  23

    Board coordinate origin is the outer top-left board corner.
    Internal corner C0 is at (1*square, 1*square, 0).
    """
    nx = squares_x - 1
    row = int(point_id) // nx
    col = int(point_id) % nx

    return np.array(
        [
            (col + 1) * square_length,
            (row + 1) * square_length,
            0.0,
        ],
        dtype=float,
    )


def rigid_transform_3d(A, B):
    """
    Solve B = R * A + t
    A: board points, shape Nx3
    B: base points, shape Nx3
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    ca = A.mean(axis=0)
    cb = B.mean(axis=0)

    AA = A - ca
    BB = B - cb

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = cb - R @ ca
    return R, t


cfg_path = Path("config/charuco_board.yaml")
pose_cam_path = Path("data/charuco_calib/charuco_pose_camera.json")
tcp_path = Path("data/charuco_calib/tcp_points_base.json")
out_path = Path("config/camera_extrinsic.yaml")

if not cfg_path.exists():
    raise SystemExit(f"Missing {cfg_path}")

if not pose_cam_path.exists():
    raise SystemExit(f"Missing {pose_cam_path}. Run detect_charuco_from_saved_frame.py first.")

if not tcp_path.exists():
    raise SystemExit(f"Missing {tcp_path}. Record TCP points first.")

cfg = yaml.safe_load(open(cfg_path))
pose_cam = json.load(open(pose_cam_path))
tcp_data = json.load(open(tcp_path))

squares_x = int(cfg["squares_x"])
square_length = float(cfg["square_length"])

point_map = {int(p["point_id"]): p for p in tcp_data["points"]}

A_board = []
B_base = []

print("Using taught points:")
for pid in cfg["teach_corner_ids"]:
    pid = int(pid)

    if pid not in point_map:
        raise SystemExit(f"Missing taught TCP point_id={pid}")

    p_board = charuco_corner_position(pid, squares_x, square_length)
    p_base = np.array(point_map[pid]["position_base"], dtype=float)

    A_board.append(p_board)
    B_base.append(p_base)

    print(f"  C{pid}: board={p_board.tolist()} -> base={p_base.tolist()}")

A_board = np.array(A_board, dtype=float)
B_base = np.array(B_base, dtype=float)

R_base_board, t_base_board = rigid_transform_3d(A_board, B_base)

pred = (R_base_board @ A_board.T).T + t_base_board
errors = np.linalg.norm(pred - B_base, axis=1)

print()
print("Teaching residuals per point [m]:", errors.tolist())
print("mean residual [m]:", float(errors.mean()))
print("max residual [m]:", float(errors.max()))

R_camera_board = np.array(
    pose_cam["T_camera_board"]["rotation_matrix"],
    dtype=float,
)
t_camera_board = np.array(
    pose_cam["T_camera_board"]["translation"],
    dtype=float,
)

# OpenCV pose:
#   p_camera = R_camera_board * p_board + t_camera_board
#
# Robot teaching:
#   p_base = R_base_board * p_board + t_base_board
#
# Therefore:
#   p_base = R_base_board * inv(R_camera_board) * (p_camera - t_camera_board) + t_base_board
#
# So:
#   R_base_camera = R_base_board * R_camera_board.T
#   t_base_camera = t_base_board - R_base_camera * t_camera_board

R_base_camera = R_base_board @ R_camera_board.T
t_base_camera = t_base_board - R_base_camera @ t_camera_board

q_base_camera = matrix_to_quat_xyzw(R_base_camera)

# Keep Ry(+90 deg) grasp-to-tcp compensation.
angle = math.radians(90.0)
q_ry90 = [
    0.0,
    math.sin(angle / 2.0),
    0.0,
    math.cos(angle / 2.0),
]

out = {
    "base_frame": cfg.get("base_frame", "base"),
    "camera_frame": cfg.get("camera_frame", "camera_color_optical_frame"),
    "T_base_camera": {
        "translation": [float(x) for x in t_base_camera],
        "quaternion_xyzw": [float(x) for x in q_base_camera],
        "rotation_matrix": [[float(v) for v in row] for row in R_base_camera],
    },
    "T_grasp_tcp": {
        "translation": [0.0, 0.0, 0.0],
        "quaternion_xyzw": [float(x) for x in q_ry90],
    },
    "calibration_source": {
        "method": "charuco_tcp_teaching",
        "board_yaml": str(cfg_path.resolve()),
        "charuco_pose_camera_json": str(pose_cam_path.resolve()),
        "tcp_points_base_json": str(tcp_path.resolve()),
        "mean_teaching_residual_m": float(errors.mean()),
        "max_teaching_residual_m": float(errors.max()),
    },
    "note": (
        "External fixed camera calibration using ChArUco board and Franka TCP-taught "
        "board corners. T_base_camera maps camera_color_optical_frame into base."
    ),
}

yaml.safe_dump(out, open(out_path, "w"), sort_keys=False)

print()
print("Computed T_base_camera:")
print(yaml.safe_dump(out, sort_keys=False))
print("Saved:", out_path)
