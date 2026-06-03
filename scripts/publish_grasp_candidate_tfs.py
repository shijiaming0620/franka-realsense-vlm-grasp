#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster


class CandidateTFPublisher(Node):
    def __init__(self, json_path, top_n, z_offset, prefix):
        super().__init__("publish_grasp_candidate_tfs")
        self.broadcaster = StaticTransformBroadcaster(self)

        data = json.load(open(json_path, "r"))
        grasps = data["grasps"]

        # 只看工作区内的，按 score 排序
        grasps = [g for g in grasps if g.get("in_workspace", True)]
        grasps = sorted(grasps, key=lambda x: x["score"], reverse=True)[:top_n]

        transforms = []
        now = self.get_clock().now().to_msg()

        for rank, g in enumerate(grasps):
            x, y, z = g["position"]
            qx, qy, qz, qw = g["orientation"]

            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = "base"
            t.child_frame_id = f"{prefix}_rank{rank:02d}_id{g['id']:03d}_s{int(g['score']*1000):04d}"

            # 抬高显示，避免埋在桌面/物体里
            t.transform.translation.x = float(x)
            t.transform.translation.y = float(y)
            t.transform.translation.z = float(z + z_offset)

            t.transform.rotation.x = float(qx)
            t.transform.rotation.y = float(qy)
            t.transform.rotation.z = float(qz)
            t.transform.rotation.w = float(qw)

            transforms.append(t)

            self.get_logger().info(
                f"{t.child_frame_id}: "
                f"pos=({x:.3f},{y:.3f},{z:.3f}), "
                f"show_z={z+z_offset:.3f}, "
                f"width={g['width']:.3f}, score={g['score']:.3f}"
            )

        self.broadcaster.sendTransform(transforms)
        self.get_logger().info(f"Published {len(transforms)} static grasp candidate TFs.")
        self.get_logger().info("Keep this terminal running while viewing RViz.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--top_n", type=int, default=15)
    parser.add_argument("--z_offset", type=float, default=0.15)
    parser.add_argument("--prefix", default="grasp")
    args = parser.parse_args()

    rclpy.init()
    node = CandidateTFPublisher(args.json, args.top_n, args.z_offset, args.prefix)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.5)
            time.sleep(0.5)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
