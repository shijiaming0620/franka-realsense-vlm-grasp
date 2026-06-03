#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/msg/robot_trajectory.hpp>

#include <franka_msgs/action/move.hpp>
#include <franka_msgs/action/grasp.hpp>

using namespace std::chrono_literals;

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

void scaleTrajectorySpeed(moveit_msgs::msg::RobotTrajectory& trajectory, double time_scale)
{
  if (time_scale <= 1.0) {
    return;
  }

  for (auto& point : trajectory.joint_trajectory.points) {
    const double t = durationToSec(point.time_from_start);
    point.time_from_start = secToDuration(t * time_scale);

    for (auto& v : point.velocities) {
      v = v / time_scale;
    }

    for (auto& a : point.accelerations) {
      a = a / (time_scale * time_scale);
    }
  }

  RCLCPP_INFO(
      rclcpp::get_logger("grasp_executor_node"),
      "Scaled trajectory time by %.2f. Motion will be slower.",
      time_scale);
}

class GraspExecutorNode : public rclcpp::Node
{
public:
  explicit GraspExecutorNode(const rclcpp::NodeOptions& options)
  : Node("grasp_executor_node", options)
  {
    sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/target_grasp_pose",
      10,
      std::bind(&GraspExecutorNode::targetPoseCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Waiting for /target_grasp_pose ...");
  }

  std::optional<geometry_msgs::msg::PoseStamped> getTargetPose()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return target_pose_;
  }

private:
  void targetPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!target_pose_) {
      target_pose_ = *msg;

      RCLCPP_INFO(
        this->get_logger(),
        "Received target pose: frame=%s, xyz=(%.3f, %.3f, %.3f), quat=(%.3f, %.3f, %.3f, %.3f)",
        msg->header.frame_id.c_str(),
        msg->pose.position.x,
        msg->pose.position.y,
        msg->pose.position.z,
        msg->pose.orientation.x,
        msg->pose.orientation.y,
        msg->pose.orientation.z,
        msg->pose.orientation.w);
    }
  }

  std::mutex mutex_;
  std::optional<geometry_msgs::msg::PoseStamped> target_pose_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_;
};

bool checkWorkspaceSafe(const geometry_msgs::msg::Pose& pose)
{
  const double x = pose.position.x;
  const double y = pose.position.y;
  const double z = pose.position.z;

  if (x < 0.25 || x > 0.80) {
    RCLCPP_ERROR(
        rclcpp::get_logger("grasp_executor_node"),
        "Target x %.3f outside safe range [0.25, 0.80].",
        x);
    return false;
  }

  if (y < -0.40 || y > 0.40) {
    RCLCPP_ERROR(
        rclcpp::get_logger("grasp_executor_node"),
        "Target y %.3f outside safe range [-0.40, 0.40].",
        y);
    return false;
  }

  if (z < -0.10 || z > 0.50) {
    RCLCPP_ERROR(
        rclcpp::get_logger("grasp_executor_node"),
        "Target z %.3f outside safe range [-0.10, 0.50].",
        z);
    return false;
  }

  return true;
}

bool executeCartesianPath(
    moveit::planning_interface::MoveGroupInterface& move_group,
    const std::vector<geometry_msgs::msg::Pose>& waypoints,
    const std::string& name,
    double time_scale,
    double min_fraction = 0.90)
{
  moveit_msgs::msg::RobotTrajectory trajectory;

  const double eef_step = 0.005;
  const double jump_threshold = 0.0;

  double fraction = move_group.computeCartesianPath(
      waypoints,
      eef_step,
      jump_threshold,
      trajectory,
      true);

  RCLCPP_INFO(
      rclcpp::get_logger("grasp_executor_node"),
      "Cartesian path %s planned %.2f%%",
      name.c_str(),
      fraction * 100.0);

  if (fraction < min_fraction) {
    RCLCPP_ERROR(
        rclcpp::get_logger("grasp_executor_node"),
        "Cartesian path fraction too low for %s. Path not executed.",
        name.c_str());
    return false;
  }

  scaleTrajectorySpeed(trajectory, time_scale);

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  plan.trajectory_ = trajectory;

  RCLCPP_INFO(
      rclcpp::get_logger("grasp_executor_node"),
      "Executing %s",
      name.c_str());

  auto result = move_group.execute(plan);

  if (result != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(
        rclcpp::get_logger("grasp_executor_node"),
        "Execution failed for %s",
        name.c_str());
    return false;
  }

  return true;
}

bool planAndExecuteJointVector(
    moveit::planning_interface::MoveGroupInterface& move_group,
    const std::vector<double>& joint_targets,
    const std::string& name,
    double time_scale)
{
  move_group.setJointValueTarget(joint_targets);

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  bool success =
      (move_group.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);

  if (!success) {
    RCLCPP_ERROR(
        rclcpp::get_logger("grasp_executor_node"),
        "Planning failed for %s",
        name.c_str());
    return false;
  }

  scaleTrajectorySpeed(plan.trajectory_, time_scale);

  RCLCPP_INFO(
      rclcpp::get_logger("grasp_executor_node"),
      "Executing %s",
      name.c_str());

  auto result = move_group.execute(plan);

  if (result != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(
        rclcpp::get_logger("grasp_executor_node"),
        "Execution failed for %s",
        name.c_str());
    return false;
  }

  return true;
}



bool planAndExecutePosition(
    moveit::planning_interface::MoveGroupInterface& move_group,
    const geometry_msgs::msg::Pose& target_pose,
    const std::string& name,
    double time_scale)
{
  move_group.setStartStateToCurrentState();

  // 只约束末端位置，不约束 orientation。
  // 用于 home -> safe_above 这种远距离安全移动，避免姿态约束导致 IK 失败。
  move_group.setPositionTarget(
      target_pose.position.x,
      target_pose.position.y,
      target_pose.position.z);

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  bool success =
      (move_group.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);

  move_group.clearPoseTargets();

  if (!success) {
    RCLCPP_ERROR(
        rclcpp::get_logger("grasp_executor_node"),
        "Position-only planning failed for %s",
        name.c_str());
    return false;
  }

  scaleTrajectorySpeed(plan.trajectory_, time_scale);

  RCLCPP_INFO(
      rclcpp::get_logger("grasp_executor_node"),
      "Executing position-only plan: %s",
      name.c_str());

  auto result = move_group.execute(plan);

  if (result != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(
        rclcpp::get_logger("grasp_executor_node"),
        "Position-only execution failed for %s",
        name.c_str());
    return false;
  }

  return true;
}

bool planAndExecutePose(
    moveit::planning_interface::MoveGroupInterface& move_group,
    const geometry_msgs::msg::Pose& target_pose,
    const std::string& name,
    double time_scale)
{
  move_group.setStartStateToCurrentState();
  move_group.setPoseTarget(target_pose);

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  bool success =
      (move_group.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);

  move_group.clearPoseTargets();

  if (!success) {
    RCLCPP_ERROR(
        rclcpp::get_logger("grasp_executor_node"),
        "Pose planning failed for %s",
        name.c_str());
    return false;
  }

  scaleTrajectorySpeed(plan.trajectory_, time_scale);

  RCLCPP_INFO(
      rclcpp::get_logger("grasp_executor_node"),
      "Executing pose plan %s",
      name.c_str());

  auto result = move_group.execute(plan);

  if (result != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(
        rclcpp::get_logger("grasp_executor_node"),
        "Pose execution failed for %s",
        name.c_str());
    return false;
  }

  return true;
}

bool openGripper(
    rclcpp::Node::SharedPtr node,
    rclcpp_action::Client<franka_msgs::action::Move>::SharedPtr client)
{
  RCLCPP_INFO(node->get_logger(), "Opening gripper...");

  if (!client->wait_for_action_server(5s)) {
    RCLCPP_ERROR(node->get_logger(), "Gripper move action server not available.");
    return false;
  }

  franka_msgs::action::Move::Goal goal;
  goal.width = 0.075;
  goal.speed = 0.03;

  auto goal_future = client->async_send_goal(goal);

  if (goal_future.wait_for(5s) != std::future_status::ready) {
    RCLCPP_ERROR(node->get_logger(), "Timeout sending open gripper goal.");
    return false;
  }

  auto goal_handle = goal_future.get();
  if (!goal_handle) {
    RCLCPP_ERROR(node->get_logger(), "Open gripper goal rejected.");
    return false;
  }

  auto result_future = client->async_get_result(goal_handle);

  if (result_future.wait_for(10s) != std::future_status::ready) {
    RCLCPP_ERROR(node->get_logger(), "Timeout waiting for open gripper result.");
    return false;
  }

  auto result = result_future.get();

  if (result.code != rclcpp_action::ResultCode::SUCCEEDED) {
    RCLCPP_ERROR(node->get_logger(), "Open gripper failed.");
    return false;
  }

  RCLCPP_INFO(node->get_logger(), "Gripper opened.");
  return true;
}

bool closeGripper(
    rclcpp::Node::SharedPtr node,
    rclcpp_action::Client<franka_msgs::action::Grasp>::SharedPtr client)
{
  RCLCPP_INFO(node->get_logger(), "Closing gripper...");

  if (!client->wait_for_action_server(5s)) {
    RCLCPP_ERROR(node->get_logger(), "Gripper grasp action server not available.");
    return false;
  }

  franka_msgs::action::Grasp::Goal goal;

  goal.width = 0.015;
  goal.speed = 0.025;
  goal.force = 30.0;
  goal.epsilon.inner = 0.02;
  goal.epsilon.outer = 0.04;

  auto goal_future = client->async_send_goal(goal);

  if (goal_future.wait_for(5s) != std::future_status::ready) {
    RCLCPP_ERROR(node->get_logger(), "Timeout sending close gripper goal.");
    return false;
  }

  auto goal_handle = goal_future.get();
  if (!goal_handle) {
    RCLCPP_ERROR(node->get_logger(), "Close gripper goal rejected.");
    return false;
  }

  auto result_future = client->async_get_result(goal_handle);

  if (result_future.wait_for(10s) != std::future_status::ready) {
    RCLCPP_ERROR(node->get_logger(), "Timeout waiting for close gripper result.");
    return false;
  }

  auto result = result_future.get();

  if (result.code != rclcpp_action::ResultCode::SUCCEEDED) {
    RCLCPP_WARN(
        node->get_logger(),
        "Close gripper did not report SUCCEEDED. Continuing.");
  } else {
    RCLCPP_INFO(node->get_logger(), "Gripper closed.");
  }

  return true;
}


geometry_msgs::msg::Quaternion normalizeQuaternion(
    const geometry_msgs::msg::Quaternion& q)
{
  geometry_msgs::msg::Quaternion out = q;
  const double n = std::sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
  if (n < 1e-12) {
    out.x = 0.0;
    out.y = 0.0;
    out.z = 0.0;
    out.w = 1.0;
    return out;
  }
  out.x /= n;
  out.y /= n;
  out.z /= n;
  out.w /= n;
  return out;
}

geometry_msgs::msg::Quaternion multiplyQuaternions(
    const geometry_msgs::msg::Quaternion& a,
    const geometry_msgs::msg::Quaternion& b)
{
  geometry_msgs::msg::Quaternion q;
  q.w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z;
  q.x = a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y;
  q.y = a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x;
  q.z = a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w;
  return normalizeQuaternion(q);
}

geometry_msgs::msg::Quaternion yawToQuaternion(double yaw)
{
  geometry_msgs::msg::Quaternion q;
  q.x = 0.0;
  q.y = 0.0;
  q.z = std::sin(yaw * 0.5);
  q.w = std::cos(yaw * 0.5);
  return q;
}

double yawFromQuaternion(const geometry_msgs::msg::Quaternion& q_in)
{
  const auto q = normalizeQuaternion(q_in);
  return std::atan2(
      2.0 * (q.w*q.z + q.x*q.y),
      1.0 - 2.0 * (q.y*q.y + q.z*q.z));
}

geometry_msgs::msg::Quaternion applyBaseYawToHomeOrientation(
    const geometry_msgs::msg::Quaternion& home_q,
    double yaw)
{
  // R_exec = Rz_base(yaw) * R_home
  // 保持 top-down 姿态，只在水平面内旋转夹爪方向。
  return multiplyQuaternions(yawToQuaternion(yaw), home_q);
}


bool executeSortingGraspSequence(
    rclcpp::Node::SharedPtr node,
    moveit::planning_interface::MoveGroupInterface& move_group,
    rclcpp_action::Client<franka_msgs::action::Move>::SharedPtr move_client,
    rclcpp_action::Client<franka_msgs::action::Grasp>::SharedPtr grasp_client,
    const std::vector<double>& home_joint_values,
    const geometry_msgs::msg::Pose& home_pose,
    const geometry_msgs::msg::Pose& grasp_pose)
{
  const double arm_time_scale = 2.5;

  // Manual click mode:
  // 只使用输入 JSON 的 position，忽略 orientation。
  // 姿态固定为 fixed home 的自然姿态，实现稳定直上直下抓取。
  geometry_msgs::msg::Pose grasp_exec_pose = grasp_pose;

  // 手动 yaw 模式：
  // grasp_pose.orientation 里只编码一个绕 base z 轴的 yaw；
  // executor 用 fixed home 姿态作为 top-down 基准，再叠加这个 yaw。
  const double manual_yaw = yawFromQuaternion(grasp_pose.orientation);
  grasp_exec_pose.orientation =
      applyBaseYawToHomeOrientation(home_pose.orientation, manual_yaw);

  RCLCPP_INFO(
      node->get_logger(),
      "Manual yaw angle: %.1f deg",
      manual_yaw * 180.0 / M_PI);

  const double approach_height = 0.22;
  const double lift_height = 0.25;
  const double place_distance_y = 0.30;   // 放置平移距离，方向根据抓取点 y 自动选择
  const double release_clearance = 0.03;

  if (!checkWorkspaceSafe(grasp_exec_pose)) {
    return false;
  }

  geometry_msgs::msg::Pose pre_grasp_pose = grasp_exec_pose;
  pre_grasp_pose.position.z += approach_height;

  geometry_msgs::msg::Pose lift_pose = grasp_exec_pose;
  lift_pose.position.z += lift_height;
  lift_pose.position.z = std::max(lift_pose.position.z, 0.20);

  geometry_msgs::msg::Pose place_above_pose = lift_pose;

  // 自动选择放置方向：
  // 抓取点在 -y 侧，就往 +y 放；
  // 抓取点在 +y 侧，就往 -y 放。
  const double place_direction_y = (grasp_exec_pose.position.y < 0.0) ? 1.0 : -1.0;
  place_above_pose.position.y += place_direction_y * place_distance_y;

  geometry_msgs::msg::Pose place_pose = place_above_pose;
  place_pose.position.z = std::max(grasp_exec_pose.position.z + release_clearance, 0.04);

  if (!checkWorkspaceSafe(pre_grasp_pose) ||
      !checkWorkspaceSafe(lift_pose) ||
      !checkWorkspaceSafe(place_above_pose) ||
      !checkWorkspaceSafe(place_pose)) {
    return false;
  }

  RCLCPP_INFO(node->get_logger(), "Manual-click fixed-orientation pick-and-place sequence:");
  RCLCPP_INFO(node->get_logger(), "  home       : x=%.3f y=%.3f z=%.3f",
              home_pose.position.x, home_pose.position.y, home_pose.position.z);
  RCLCPP_INFO(node->get_logger(), "  pre_grasp  : x=%.3f y=%.3f z=%.3f  orientation=fixed_home",
              pre_grasp_pose.position.x, pre_grasp_pose.position.y, pre_grasp_pose.position.z);
  RCLCPP_INFO(node->get_logger(), "  grasp      : x=%.3f y=%.3f z=%.3f  orientation=fixed_home",
              grasp_exec_pose.position.x, grasp_exec_pose.position.y, grasp_exec_pose.position.z);
  RCLCPP_INFO(node->get_logger(), "  lift       : x=%.3f y=%.3f z=%.3f  orientation=fixed_home",
              lift_pose.position.x, lift_pose.position.y, lift_pose.position.z);
  RCLCPP_INFO(node->get_logger(), "  place_above: x=%.3f y=%.3f z=%.3f  orientation=fixed_home",
              place_above_pose.position.x, place_above_pose.position.y, place_above_pose.position.z);
  RCLCPP_INFO(node->get_logger(), "  place      : x=%.3f y=%.3f z=%.3f  orientation=fixed_home",
              place_pose.position.x, place_pose.position.y, place_pose.position.z);

  RCLCPP_INFO(node->get_logger(), "Step 0: open gripper.");
  if (!openGripper(node, move_client)) {
    return false;
  }

  std::this_thread::sleep_for(300ms);

  RCLCPP_INFO(node->get_logger(), "Step 1: move to pre_grasp with fixed home orientation.");
  if (!planAndExecutePose(
          move_group,
          pre_grasp_pose,
          "manual_move_to_pre_grasp_fixed_orientation",
          arm_time_scale)) {
    return false;
  }

  std::this_thread::sleep_for(300ms);

  RCLCPP_INFO(node->get_logger(), "Step 2: Cartesian descend to grasp.");
  if (!executeCartesianPath(
          move_group,
          std::vector<geometry_msgs::msg::Pose>{grasp_exec_pose},
          "manual_pre_grasp_to_grasp",
          arm_time_scale)) {
    return false;
  }

  std::this_thread::sleep_for(500ms);

  RCLCPP_INFO(node->get_logger(), "Step 3: close gripper.");
  if (!closeGripper(node, grasp_client)) {
    return false;
  }

  std::this_thread::sleep_for(800ms);

  RCLCPP_INFO(node->get_logger(), "Step 4: Cartesian lift object.");
  if (!executeCartesianPath(
          move_group,
          std::vector<geometry_msgs::msg::Pose>{lift_pose},
          "manual_lift_after_grasp",
          arm_time_scale)) {
    return false;
  }

  std::this_thread::sleep_for(300ms);

  RCLCPP_INFO(node->get_logger(), "Step 5: Cartesian move to place_above.");
  if (!executeCartesianPath(
          move_group,
          std::vector<geometry_msgs::msg::Pose>{place_above_pose},
          "manual_lift_to_place_above",
          arm_time_scale,
          0.80)) {
    RCLCPP_WARN(node->get_logger(), "Cartesian move to place_above failed. Trying pose planning.");
    if (!planAndExecutePose(
            move_group,
            place_above_pose,
            "manual_lift_to_place_above_pose_plan",
            arm_time_scale)) {
      return false;
    }
  }

  std::this_thread::sleep_for(300ms);

  RCLCPP_INFO(node->get_logger(), "Step 6: Cartesian descend to place.");
  if (!executeCartesianPath(
          move_group,
          std::vector<geometry_msgs::msg::Pose>{place_pose},
          "manual_place_above_to_place",
          arm_time_scale)) {
    return false;
  }

  std::this_thread::sleep_for(300ms);

  RCLCPP_INFO(node->get_logger(), "Step 7: open gripper to release.");
  if (!openGripper(node, move_client)) {
    return false;
  }

  std::this_thread::sleep_for(500ms);

  RCLCPP_INFO(node->get_logger(), "Step 8: Cartesian lift after release.");
  if (!executeCartesianPath(
          move_group,
          std::vector<geometry_msgs::msg::Pose>{place_above_pose},
          "manual_lift_after_release",
          arm_time_scale)) {
    return false;
  }

  std::this_thread::sleep_for(300ms);

  RCLCPP_INFO(node->get_logger(), "Step 9: return to fixed home joint pose.");
  if (!planAndExecuteJointVector(
          move_group,
          home_joint_values,
          "manual_return_fixed_home_after_sorting",
          arm_time_scale)) {
    return false;
  }

  RCLCPP_INFO(node->get_logger(), "Manual-click pick-and-place finished.");
  return true;
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions node_options;
  node_options.automatically_declare_parameters_from_overrides(true);

  auto node = std::make_shared<GraspExecutorNode>(node_options);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor]() { executor.spin(); });

  auto move_client =
      rclcpp_action::create_client<franka_msgs::action::Move>(
          node,
          "/franka_gripper/move");

  auto grasp_client =
      rclcpp_action::create_client<franka_msgs::action::Grasp>(
          node,
          "/franka_gripper/grasp");

  moveit::planning_interface::MoveGroupInterface move_group(node, "fr3_arm");

  move_group.setMaxVelocityScalingFactor(0.18);
  move_group.setMaxAccelerationScalingFactor(0.15);
  move_group.setPlanningTime(30.0);
  move_group.setNumPlanningAttempts(30);

  RCLCPP_INFO(node->get_logger(), "Planning group: fr3_arm");
  RCLCPP_INFO(
      node->get_logger(),
      "End effector link: %s",
      move_group.getEndEffectorLink().c_str());

  std::vector<double> home_joint_values = { 0.002425678074, -0.967908084393, -0.030450243503, -1.765445351601, -0.011149937287, 0.841378748417, 0.730756044388 };

  RCLCPP_INFO(node->get_logger(), "Moving to fixed home joint pose first.");
  if (!planAndExecuteJointVector(
          move_group,
          home_joint_values,
          "move_to_fixed_home_at_start",
          4.0)) {
    RCLCPP_ERROR(node->get_logger(), "Failed to move to fixed home at startup.");
    rclcpp::shutdown();
    spinner.join();
    return 1;
  }

  geometry_msgs::msg::PoseStamped home_pose_stamped = move_group.getCurrentPose();

  RCLCPP_INFO(
      node->get_logger(),
      "Fixed home pose reached: position=(%.3f, %.3f, %.3f)",
      home_pose_stamped.pose.position.x,
      home_pose_stamped.pose.position.y,
      home_pose_stamped.pose.position.z);

  std::optional<geometry_msgs::msg::PoseStamped> target_pose_stamped;

  rclcpp::Rate rate(10.0);
  const auto start_time = node->now();

  while (rclcpp::ok()) {
    target_pose_stamped = node->getTargetPose();

    if (target_pose_stamped) {
      break;
    }

    if ((node->now() - start_time).seconds() > 30.0) {
      RCLCPP_ERROR(node->get_logger(), "Timeout waiting for /target_grasp_pose.");
      rclcpp::shutdown();
      spinner.join();
      return 1;
    }

    rate.sleep();
  }

  if (target_pose_stamped->header.frame_id != "base") {
    RCLCPP_WARN(
      node->get_logger(),
      "Target pose frame_id is '%s'. This demo expects frame_id='base'.",
      target_pose_stamped->header.frame_id.c_str());
  }

  bool ok = executeSortingGraspSequence(
      node,
      move_group,
      move_client,
      grasp_client,
      home_joint_values,
      home_pose_stamped.pose,
      target_pose_stamped->pose);

  rclcpp::shutdown();
  spinner.join();

  return ok ? 0 : 1;
}
