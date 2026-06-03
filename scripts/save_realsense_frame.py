#!/usr/bin/env python3
import json
import os
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


class RealSenseFrameSaver(Node):
    def __init__(self):
        super().__init__("save_realsense_frame")

        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("save_dir", str(Path.home() / "franka_grasp_ws/src/franka_grasp_demo/data/realsense_sample"))

        self.color_topic = self.get_parameter("color_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.save_dir = Path(self.get_parameter("save_dir").value)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.bridge = CvBridge()

        self.color_msg = None
        self.depth_msg = None
        self.camera_info_msg = None
        self.saved = False

        self.create_subscription(Image, self.color_topic, self.color_cb, 10)
        self.create_subscription(Image, self.depth_topic, self.depth_cb, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.info_cb, 10)

        self.timer = self.create_timer(0.5, self.try_save)

        self.get_logger().info(f"Waiting for color: {self.color_topic}")
        self.get_logger().info(f"Waiting for depth: {self.depth_topic}")
        self.get_logger().info(f"Waiting for camera_info: {self.camera_info_topic}")

    def color_cb(self, msg):
        self.color_msg = msg

    def depth_cb(self, msg):
        self.depth_msg = msg

    def info_cb(self, msg):
        self.camera_info_msg = msg

    def try_save(self):
        if self.saved:
            return

        if self.color_msg is None or self.depth_msg is None or self.camera_info_msg is None:
            self.get_logger().info("Waiting for all messages...")
            return

        color = self.bridge.imgmsg_to_cv2(self.color_msg, desired_encoding="bgr8")
        depth = self.bridge.imgmsg_to_cv2(self.depth_msg, desired_encoding="passthrough")

        color_path = self.save_dir / "color.png"
        depth_path = self.save_dir / "depth.png"
        info_path = self.save_dir / "camera_info.json"

        cv2.imwrite(str(color_path), color)
        cv2.imwrite(str(depth_path), depth)

        info = {
            "width": self.camera_info_msg.width,
            "height": self.camera_info_msg.height,
            "k": list(self.camera_info_msg.k),
            "d": list(self.camera_info_msg.d),
            "distortion_model": self.camera_info_msg.distortion_model,
            "frame_id": self.camera_info_msg.header.frame_id,
        }

        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

        self.saved = True
        self.get_logger().info(f"Saved color to {color_path}")
        self.get_logger().info(f"Saved depth to {depth_path}")
        self.get_logger().info(f"Saved camera info to {info_path}")
        self.get_logger().info("Done. You can Ctrl+C now.")


def main():
    rclpy.init()
    node = RealSenseFrameSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
