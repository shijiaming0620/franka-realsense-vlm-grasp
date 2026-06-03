import os
import yaml

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    with open(absolute_file_path, "r") as file:
        return yaml.safe_load(file)


def generate_launch_description():
    robot_ip = LaunchConfiguration("robot_ip")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    fake_sensor_commands = LaunchConfiguration("fake_sensor_commands")
    load_gripper = LaunchConfiguration("load_gripper")
    ee_id = LaunchConfiguration("ee_id")

    franka_xacro_file = os.path.join(
        get_package_share_directory("franka_description"),
        "robots",
        "fr3",
        "fr3.urdf.xacro",
    )

    robot_description_config = Command(
        [
            FindExecutable(name="xacro"),
            " ",
            franka_xacro_file,
            " hand:=",
            load_gripper,
            " robot_ip:=",
            robot_ip,
            " ee_id:=",
            ee_id,
            " use_fake_hardware:=",
            use_fake_hardware,
            " fake_sensor_commands:=",
            fake_sensor_commands,
            " ros2_control:=true",
        ]
    )

    robot_description = {
        "robot_description": ParameterValue(
            robot_description_config,
            value_type=str,
        )
    }

    franka_semantic_xacro_file = os.path.join(
        get_package_share_directory("franka_description"),
        "robots",
        "fr3",
        "fr3.srdf.xacro",
    )

    robot_description_semantic_config = Command(
        [
            FindExecutable(name="xacro"),
            " ",
            franka_semantic_xacro_file,
            " hand:=",
            load_gripper,
            " ee_id:=",
            ee_id,
        ]
    )

    robot_description_semantic = {
        "robot_description_semantic": ParameterValue(
            robot_description_semantic_config,
            value_type=str,
        )
    }

    kinematics_yaml = load_yaml(
        "franka_fr3_moveit_config",
        "config/kinematics.yaml",
    )

    grasp_executor_node = Node(
        package="franka_grasp_demo",
        executable="grasp_executor_node",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_ip", default_value="dont-care"),
            DeclareLaunchArgument("use_fake_hardware", default_value="true"),
            DeclareLaunchArgument("fake_sensor_commands", default_value="false"),
            DeclareLaunchArgument("load_gripper", default_value="true"),
            DeclareLaunchArgument("ee_id", default_value="franka_hand"),
            grasp_executor_node,
        ]
    )
