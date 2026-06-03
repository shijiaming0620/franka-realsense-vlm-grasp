#!/usr/bin/env python3
import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class GraspNetBridgeNode(Node):
    def __init__(self):
        super().__init__("graspnet_bridge_node")

        default_json = (
            Path.home()
            / "franka_grasp_ws/src/franka_grasp_demo/data/graspnet_result/grasp_result.json"
        )

        self.declare_parameter("json_path", str(default_json))
        self.declare_parameter("publish_once", True)
        self.declare_parameter("publish_rate", 1.0)

        self.json_path = Path(self.get_parameter("json_path").value)
        self.publish_once = bool(self.get_parameter("publish_once").value)
        self.publish_rate = float(self.get_parameter("publish_rate").value)

        self.pub = self.create_publisher(PoseStamped, "/target_grasp_pose", 10)

        self.has_published = False
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_grasp)

        self.get_logger().info(f"Reading grasp result from: {self.json_path}")
        self.get_logger().info("Publishing to /target_grasp_pose")

    def load_grasp(self):
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")

        with open(self.json_path, "r") as f:
            data = json.load(f)

        frame_id = data.get("frame_id", "base")
        position = data["position"]
        orientation = data["orientation"]

        if len(position) != 3:
            raise ValueError("position must be [x, y, z]")

        if len(orientation) != 4:
            raise ValueError("orientation must be [qx, qy, qz, qw]")

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])

        msg.pose.orientation.x = float(orientation[0])
        msg.pose.orientation.y = float(orientation[1])
        msg.pose.orientation.z = float(orientation[2])
        msg.pose.orientation.w = float(orientation[3])

        return msg, data

    def publish_grasp(self):
        if self.publish_once and self.has_published:
            return

        try:
            msg, data = self.load_grasp()
        except Exception as e:
            self.get_logger().error(f"Failed to load grasp JSON: {e}")
            return

        self.pub.publish(msg)
        self.has_published = True

        score = data.get("score", None)
        width = data.get("width", None)

        self.get_logger().info(
            "Published /target_grasp_pose: "
            f"frame={msg.header.frame_id}, "
            f"xyz=({msg.pose.position.x:.3f}, "
            f"{msg.pose.position.y:.3f}, "
            f"{msg.pose.position.z:.3f}), "
            f"quat=({msg.pose.orientation.x:.3f}, "
            f"{msg.pose.orientation.y:.3f}, "
            f"{msg.pose.orientation.z:.3f}, "
            f"{msg.pose.orientation.w:.3f}), "
            f"score={score}, width={width}"
        )

        if self.publish_once:
            self.get_logger().info("Published once. You can Ctrl+C now.")


def main():
    rclpy.init()
    node = GraspNetBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
