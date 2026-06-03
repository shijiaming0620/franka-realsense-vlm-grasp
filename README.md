# Franka RealSense VLM Grasp Demo

A ROS 2 Humble robotic grasping demo for Franka FR3 + Intel RealSense.

## Features

- RealSense RGB-D online image acquisition
- Manual grasp point selection
- Manual yaw direction selection
- ChArUco-based external camera calibration
- VLM-assisted multi-object proposal
- Experimental depth/geometric refinement
- MoveIt-based Franka FR3 execution

## Manual Online Grasp

```bash
python3 scripts/online_click_execute_manual_grasp_yaw.py \
  --z_offset 0.030 \
  --depth_radius 5
