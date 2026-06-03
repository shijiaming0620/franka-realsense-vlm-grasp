#!/usr/bin/env python3
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class WebcamPublisher(Node):
    def __init__(self):
        super().__init__("webcam_publisher")

        self.declare_parameter("camera_index", 0)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("topic", "/webcam/image_raw")

        camera_index = int(self.get_parameter("camera_index").value)
        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        fps = float(self.get_parameter("fps").value)
        topic = str(self.get_parameter("topic").value)

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, topic, 10)
        self.timer = self.create_timer(1.0 / fps, self.publish_frame)

        self.get_logger().info(f"Publishing webcam to {topic}")

    def publish_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn("Failed to read frame from camera")
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "webcam_frame"
        self.pub.publish(msg)

    def destroy_node(self):
        if hasattr(self, "cap"):
            self.cap.release()
        super().destroy_node()


def main():
    rclpy.init()
    node = WebcamPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
