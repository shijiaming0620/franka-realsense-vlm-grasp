cmake_minimum_required(VERSION 3.8)
project(franka_grasp_demo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(franka_msgs REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

# 旧的测试节点：你之前用来跑固定轨迹的节点
add_executable(fake_grasp_node
  src/fake_grasp_node.cpp
)

ament_target_dependencies(fake_grasp_node
  rclcpp
  geometry_msgs
  moveit_ros_planning_interface
  moveit_msgs
  trajectory_msgs
)

# 新的执行节点：后面接 /target_grasp_pose
add_executable(grasp_executor_node
  src/grasp_executor_node.cpp
)

ament_target_dependencies(grasp_executor_node
  rclcpp
  geometry_msgs
  moveit_ros_planning_interface
  moveit_msgs
  trajectory_msgs
  rclcpp_action
  franka_msgs
)

# 新的假抓取姿态发布节点：模拟 GraspNet 输出
add_executable(fake_grasp_pose_publisher
  src/fake_grasp_pose_publisher.cpp
)

ament_target_dependencies(fake_grasp_pose_publisher
  rclcpp
  geometry_msgs
)

install(TARGETS
  fake_grasp_node
  grasp_executor_node
  fake_grasp_pose_publisher
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/webcam_publisher.py
  scripts/save_realsense_frame.py
  scripts/transform_grasp_camera_to_base.py
  scripts/graspnet_bridge_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
