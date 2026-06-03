#!/usr/bin/env python3
import json
from pathlib import Path

import cv2
import numpy as np


def load_camera_info(path):
    with open(path, "r") as f:
        info = json.load(f)

    k = np.array(info["k"], dtype=np.float64).reshape(3, 3)
    d = np.array(info["d"], dtype=np.float64)

    frame_id = info.get("frame_id", "camera_color_optical_frame")
    return k, d, frame_id


def rvec_to_quat_xyzw(rvec):
    rmat, _ = cv2.Rodrigues(rvec)
    trace = np.trace(rmat)

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rmat[2, 1] - rmat[1, 2]) / s
        qy = (rmat[0, 2] - rmat[2, 0]) / s
        qz = (rmat[1, 0] - rmat[0, 1]) / s
    elif rmat[0, 0] > rmat[1, 1] and rmat[0, 0] > rmat[2, 2]:
        s = np.sqrt(1.0 + rmat[0, 0] - rmat[1, 1] - rmat[2, 2]) * 2.0
        qw = (rmat[2, 1] - rmat[1, 2]) / s
        qx = 0.25 * s
        qy = (rmat[0, 1] + rmat[1, 0]) / s
        qz = (rmat[0, 2] + rmat[2, 0]) / s
    elif rmat[1, 1] > rmat[2, 2]:
        s = np.sqrt(1.0 + rmat[1, 1] - rmat[0, 0] - rmat[2, 2]) * 2.0
        qw = (rmat[0, 2] - rmat[2, 0]) / s
        qx = (rmat[0, 1] + rmat[1, 0]) / s
        qy = 0.25 * s
        qz = (rmat[1, 2] + rmat[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rmat[2, 2] - rmat[0, 0] - rmat[1, 1]) * 2.0
        qw = (rmat[1, 0] - rmat[0, 1]) / s
        qx = (rmat[0, 2] + rmat[2, 0]) / s
        qy = (rmat[1, 2] + rmat[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q = q / np.linalg.norm(q)
    return q, rmat


def main():
    data_dir = Path.home() / "franka_grasp_ws/src/franka_grasp_demo/data/external_aruco_calib_8cm"

    color_path = data_dir / "color.png"
    camera_info_path = data_dir / "camera_info.json"
    output_json = data_dir / "aruco_pose_camera.json"
    output_vis = data_dir / "aruco_detected.png"

    marker_size = 0.08
    expected_id = 0

    image = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {color_path}")

    camera_matrix, dist_coeffs, camera_frame = load_camera_info(camera_info_path)

    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco not found. Install OpenCV contrib package first."
        )

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if hasattr(aruco, "ArucoDetector"):
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=parameters)

    if ids is None or len(ids) == 0:
        raise RuntimeError("No ArUco marker detected.")

    ids_flat = ids.flatten().tolist()
    print("Detected marker ids:", ids_flat)

    if expected_id not in ids_flat:
        raise RuntimeError(f"Expected marker id {expected_id}, but detected {ids_flat}")

    idx = ids_flat.index(expected_id)
    marker_corners = corners[idx].reshape(4, 2).astype(np.float64)

    half = marker_size / 2.0

    # OpenCV 检测角点顺序：
    # top-left, top-right, bottom-right, bottom-left
    object_points = np.array(
        [
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        marker_corners,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )

    if not success:
        raise RuntimeError("solvePnP failed.")

    q_xyzw, rmat = rvec_to_quat_xyzw(rvec)

    t = tvec.reshape(3)

    result = {
        "camera_frame": camera_frame,
        "marker_frame": "aruco_marker",
        "marker_id": expected_id,
        "marker_size": marker_size,
        "T_camera_marker": {
            "translation": [
                float(t[0]),
                float(t[1]),
                float(t[2]),
            ],
            "quaternion_xyzw": [
                float(q_xyzw[0]),
                float(q_xyzw[1]),
                float(q_xyzw[2]),
                float(q_xyzw[3]),
            ],
            "rotation_matrix": rmat.tolist(),
        },
        "note": "T_camera_marker maps marker coordinates into camera_color_optical_frame."
    }

    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)

    vis = image.copy()
    aruco.drawDetectedMarkers(vis, corners, ids)

    try:
        cv2.drawFrameAxes(
            vis,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
            marker_size * 0.5,
        )
    except Exception:
        pass

    cv2.imwrite(str(output_vis), vis)

    print(json.dumps(result, indent=2))
    print(f"Saved pose json: {output_json}")
    print(f"Saved visualization: {output_vis}")


if __name__ == "__main__":
    main()
