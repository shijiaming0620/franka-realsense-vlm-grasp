import argparse
import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

import tf2_ros


class Recorder(Node):
    def __init__(self):
        super().__init__("record_charuco_tcp_point")
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer, self)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--point_id", type=int, required=True)
    parser.add_argument("--base_frame", default="base")
    parser.add_argument("--tcp_frame", default="fr3_hand_tcp")
    parser.add_argument("--out", default="data/charuco_calib/tcp_points_base.json")
    args = parser.parse_args()

    rclpy.init()
    node = Recorder()

    print(f"Recording point_id={args.point_id}")
    print(f"Looking up TF: {args.base_frame} -> {args.tcp_frame}")

    tf = None
    for _ in range(50):
        try:
            tf = node.buffer.lookup_transform(
                args.base_frame,
                args.tcp_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )
            break
        except Exception:
            rclpy.spin_once(node, timeout_sec=0.1)

    if tf is None:
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit("Failed to lookup transform. Is robot/MoveIt running?")

    tr = tf.transform.translation
    q = tf.transform.rotation

    item = {
        "point_id": int(args.point_id),
        "frame_id": args.base_frame,
        "tcp_frame": args.tcp_frame,
        "position_base": [float(tr.x), float(tr.y), float(tr.z)],
        "orientation_xyzw": [float(q.x), float(q.y), float(q.z), float(q.w)],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        data = json.load(open(out))
    else:
        data = {"points": []}

    data["points"] = [p for p in data["points"] if int(p["point_id"]) != args.point_id]
    data["points"].append(item)
    data["points"] = sorted(data["points"], key=lambda x: int(x["point_id"]))

    json.dump(data, open(out, "w"), indent=2)

    print("Saved point:")
    print(json.dumps(item, indent=2))
    print("Updated:", out)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
