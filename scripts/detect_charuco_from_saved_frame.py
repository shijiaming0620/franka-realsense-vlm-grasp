import json
import yaml
import cv2
import numpy as np
from pathlib import Path
import math

def matrix_to_quat_xyzw(R):
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    q = np.array([qx, qy, qz, qw], dtype=float)
    q = q / np.linalg.norm(q)
    return q.tolist()

cfg = yaml.safe_load(open("config/charuco_board.yaml"))

base_dir = Path("data/charuco_calib")
color_path = base_dir / "color.png"
info_path = base_dir / "camera_info.json"
out_json = base_dir / "charuco_pose_camera.json"
out_vis = base_dir / "charuco_detected.png"

dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg["dictionary"]))

board = cv2.aruco.CharucoBoard_create(
    int(cfg["squares_x"]),
    int(cfg["squares_y"]),
    float(cfg["square_length"]),
    float(cfg["marker_length"]),
    dictionary
)

img = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
if img is None:
    raise SystemExit(f"Failed to read {color_path}")

info = json.load(open(info_path))
K_raw = info.get("k") or info.get("K")
D_raw = info.get("d") or info.get("D") or [0, 0, 0, 0, 0]

K = np.array(K_raw, dtype=float).reshape(3, 3)
D = np.array(D_raw, dtype=float).reshape(-1, 1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary)

if ids is None or len(ids) == 0:
    raise SystemExit("No ArUco markers detected on ChArUco board.")

ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
    corners,
    ids,
    gray,
    board,
    K,
    D
)

if charuco_ids is None or ret < 4:
    raise SystemExit(f"Not enough ChArUco corners. ret={ret}")

rvec = np.zeros((3, 1), dtype=float)
tvec = np.zeros((3, 1), dtype=float)

ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
    charuco_corners,
    charuco_ids,
    board,
    K,
    D,
    rvec,
    tvec
)

if not ok:
    raise SystemExit("estimatePoseCharucoBoard failed.")

R, _ = cv2.Rodrigues(rvec)
t = tvec.reshape(3)
q = matrix_to_quat_xyzw(R)

data = {
    "frame_id": cfg["camera_frame"],
    "board_frame": cfg["board_frame"],
    "T_camera_board": {
        "translation": [float(x) for x in t],
        "quaternion_xyzw": [float(x) for x in q],
        "rotation_matrix": [[float(v) for v in row] for row in R],
    },
    "num_charuco_corners": int(ret),
    "charuco_ids": [int(x) for x in charuco_ids.flatten().tolist()],
    "note": "T_camera_board maps board coordinates into camera_color_optical_frame."
}

json.dump(data, open(out_json, "w"), indent=2)

vis = img.copy()
cv2.aruco.drawDetectedMarkers(vis, corners, ids)
cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)

try:
    cv2.aruco.drawAxis(vis, K, D, rvec, tvec, 0.08)
except Exception:
    pass

cv2.imwrite(str(out_vis), vis)

print("Detected ChArUco corners:", int(ret))
print("Saved:", out_json)
print("Saved:", out_vis)
print("T_camera_board translation:", t.tolist())
print("T_camera_board quaternion xyzw:", q)
