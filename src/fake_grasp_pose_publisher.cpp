#include <chrono>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

using namespace std::chrono_literals;

class FakeGraspPosePublisher : public rclcpp::Node
{
public:
  FakeGraspPosePublisher() : Node("fake_grasp_pose_publisher")
  {
    pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
      "/target_grasp_pose", 10);

    timer_ = this->create_wall_timer(
      1s, std::bind(&FakeGraspPosePublisher::publishPose, this));

    RCLCPP_INFO(this->get_logger(), "Fake grasp pose publisher started.");
  }

private:
  void publishPose()
  {
    geometry_msgs::msg::PoseStamped msg;
    msg.header.stamp = this->now();
    msg.header.frame_id = "base";

    // 目前先模拟：目标在绿色 box 正上方
    msg.pose.position.x = 0.45;
    msg.pose.position.y = 0.00;
    msg.pose.position.z = 0.25;

    // 先用当前你仿真里比较稳定的 TCP 姿态
    // 后面 GraspNet 接入后，这里会换成 GraspNet 输出的 orientation
    msg.pose.orientation.x = 1.0;
    msg.pose.orientation.y = 0.0;
    msg.pose.orientation.z = 0.0;
    msg.pose.orientation.w = 0.0;

    pub_->publish(msg);

    RCLCPP_INFO(
      this->get_logger(),
      "Published fake target grasp pose: frame=%s, xyz=(%.3f, %.3f, %.3f)",
      msg.header.frame_id.c_str(),
      msg.pose.position.x,
      msg.pose.position.y,
      msg.pose.position.z);
  }

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FakeGraspPosePublisher>());
  rclcpp::shutdown();
  return 0;
}
