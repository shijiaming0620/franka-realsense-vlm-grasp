#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>


double durationToSec(const builtin_interfaces::msg::Duration& d)
{
  return static_cast<double>(d.sec) + static_cast<double>(d.nanosec) * 1e-9;
}


builtin_interfaces::msg::Duration secToDuration(double t)
{
  builtin_interfaces::msg::Duration d;
  d.sec = static_cast<int32_t>(std::floor(t));
  d.nanosec = static_cast<uint32_t>((t - std::floor(t)) * 1e9);
  return d;
}


void addSecondsToPoint(trajectory_msgs::msg::JointTrajectoryPoint& p, double seconds)
{
  double t = durationToSec(p.time_from_start);
  p.time_from_start = secToDuration(t + seconds);
}


double poseDistance(
    const geometry_msgs::msg::Pose& a,
    const geometry_msgs::msg::Pose& b)
{
  const double dx = a.position.x - b.position.x;
  const double dy = a.position.y - b.position.y;
  const double dz = a.position.z - b.position.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}


void zeroVelAcc(trajectory_msgs::msg::JointTrajectoryPoint& p)
{
  if (!p.velocities.empty()) {
    std::fill(p.velocities.begin(), p.velocities.end(), 0.0);
  }
  if (!p.accelerations.empty()) {
    std::fill(p.accelerations.begin(), p.accelerations.end(), 0.0);
  }
  if (!p.effort.empty()) {
    std::fill(p.effort.begin(), p.effort.end(), 0.0);
  }
}


bool insertPauseAtIndex(
    moveit_msgs::msg::RobotTrajectory& trajectory,
    std::size_t pause_index,
    double pause_seconds)
{
  auto& points = trajectory.joint_trajectory.points;

  if (points.empty()) {
    RCLCPP_ERROR(rclcpp::get_logger("one_piece_motion_node"), "Trajectory has no points.");
    return false;
  }

  if (pause_index >= points.size()) {
    pause_index = points.size() - 1;
  }

  trajectory_msgs::msg::JointTrajectoryPoint pause_start = points[pause_index];
  zeroVelAcc(pause_start);

  trajectory_msgs::msg::JointTrajectoryPoint pause_end = pause_start;
  addSecondsToPoint(pause_end, pause_seconds);

  points[pause_index] = pause_start;
  points.insert(points.begin() + pause_index + 1, pause_end);

  for (std::size_t i = pause_index + 2; i < points.size(); ++i) {
    addSecondsToPoint(points[i], pause_seconds);
  }

  RCLCPP_INFO(
      rclcpp::get_logger("one_piece_motion_node"),
      "Inserted %.2f s pause at trajectory point %zu / %zu",
      pause_seconds,
      pause_index,
      points.size());

  return true;
}


bool executeOneCartesianTrajectoryWithPause(
    moveit::planning_interface::MoveGroupInterface& move_group,
    const geometry_msgs::msg::Pose& home_pose,
    const geometry_msgs::msg::Pose& above_box_pose,
    const geometry_msgs::msg::Pose& lift_pose,
    const geometry_msgs::msg::Pose& side_pose)
{
  moveit_msgs::msg::RobotTrajectory trajectory;

  const double eef_step = 0.01;
  const double jump_threshold = 0.0;

  std::vector<geometry_msgs::msg::Pose> waypoints;
  waypoints.push_back(above_box_pose);
  waypoints.push_back(lift_pose);
  waypoints.push_back(side_pose);
  waypoints.push_back(home_pose);

  double fraction = move_group.computeCartesianPath(
      waypoints,
      eef_step,
      jump_threshold,
      trajectory,
      true);

  RCLCPP_INFO(
      rclcpp::get_logger("one_piece_motion_node"),
      "Cartesian path planned %.2f%%",
      fraction * 100.0);

  if (fraction < 0.90) {
    RCLCPP_ERROR(
        rclcpp::get_logger("one_piece_motion_node"),
        "Cartesian path fraction too low. Path not executed.");
    return false;
  }

  auto& points = trajectory.joint_trajectory.points;
  if (points.empty()) {
    RCLCPP_ERROR(
        rclcpp::get_logger("one_piece_motion_node"),
        "Computed trajectory has no points.");
    return false;
  }

  // 估算“到达物体上方”的轨迹点索引。
  // 第一段是 home -> above_box，采样间隔 eef_step。
  const double first_segment_distance = poseDistance(home_pose, above_box_pose);
  std::size_t pause_index =
      static_cast<std::size_t>(std::ceil(first_segment_distance / eef_step));

  if (pause_index >= points.size()) {
    pause_index = points.size() / 4;
  }

  // 在物体上方插入 2 秒停顿。
  if (!insertPauseAtIndex(trajectory, pause_index, 2.0)) {
    return false;
  }

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  plan.trajectory_ = trajectory;

  RCLCPP_INFO(
      rclcpp::get_logger("one_piece_motion_node"),
      "Executing one complete trajectory with 2-second pause...");

  auto result = move_group.execute(plan);

  if (result != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(
        rclcpp::get_logger("one_piece_motion_node"),
        "Trajectory execution failed.");
    return false;
  }

  RCLCPP_INFO(
      rclcpp::get_logger("one_piece_motion_node"),
      "One complete trajectory finished.");

  return true;
}


int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions node_options;
  node_options.automatically_declare_parameters_from_overrides(true);

  auto node = rclcpp::Node::make_shared("one_piece_motion_node", node_options);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor]() { executor.spin(); });

  const std::string planning_group = "fr3_arm";

  moveit::planning_interface::MoveGroupInterface move_group(node, planning_group);

  move_group.setMaxVelocityScalingFactor(0.10);
  move_group.setMaxAccelerationScalingFactor(0.10);
  move_group.setPlanningTime(10.0);
  move_group.setNumPlanningAttempts(10);

  RCLCPP_INFO(node->get_logger(), "Planning group: %s", planning_group.c_str());
  RCLCPP_INFO(
      node->get_logger(),
      "End effector link: %s",
      move_group.getEndEffectorLink().c_str());

  geometry_msgs::msg::PoseStamped home_pose_stamped = move_group.getCurrentPose();
  geometry_msgs::msg::Pose home_pose = home_pose_stamped.pose;

  RCLCPP_INFO(
      node->get_logger(),
      "Saved home pose: position=(%.3f, %.3f, %.3f)",
      home_pose.position.x,
      home_pose.position.y,
      home_pose.position.z);

  const double box_x = 0.45;
  const double box_y = 0.00;

  const double above_z = 0.25;
  const double lift_distance = 0.15;
  const double side_distance = 0.20;

  geometry_msgs::msg::Pose above_box_pose = home_pose;
  above_box_pose.position.x = box_x;
  above_box_pose.position.y = box_y;
  above_box_pose.position.z = above_z;

  geometry_msgs::msg::Pose lift_pose = above_box_pose;
  lift_pose.position.z = above_z + lift_distance;

  geometry_msgs::msg::Pose side_pose = lift_pose;
  side_pose.position.y = box_y + side_distance;

  RCLCPP_INFO(node->get_logger(), "One trajectory with pause:");
  RCLCPP_INFO(
      node->get_logger(),
      "  1. Move above box: x=%.3f y=%.3f z=%.3f",
      above_box_pose.position.x,
      above_box_pose.position.y,
      above_box_pose.position.z);

  RCLCPP_INFO(node->get_logger(), "  2. Pause 2 seconds above box");

  RCLCPP_INFO(
      node->get_logger(),
      "  3. Lift: x=%.3f y=%.3f z=%.3f",
      lift_pose.position.x,
      lift_pose.position.y,
      lift_pose.position.z);

  RCLCPP_INFO(
      node->get_logger(),
      "  4. Move side: x=%.3f y=%.3f z=%.3f",
      side_pose.position.x,
      side_pose.position.y,
      side_pose.position.z);

  RCLCPP_INFO(node->get_logger(), "  5. Return home");

  bool ok = executeOneCartesianTrajectoryWithPause(
      move_group,
      home_pose,
      above_box_pose,
      lift_pose,
      side_pose);

  rclcpp::shutdown();
  spinner.join();

  return ok ? 0 : 1;
}