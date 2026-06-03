#!/usr/bin/env bash
set -e

cd ~/franka_grasp_ws/src/franka_grasp_demo

conda deactivate 2>/dev/null || true
unset LD_LIBRARY_PATH
unset LIBRARY_PATH
unset CPATH
unset PKG_CONFIG_PATH

source /opt/ros/humble/setup.bash
source ~/franka_ros2_ws/install/setup.bash
source ~/franka_grasp_ws/install/setup.bash

MANUAL_JSON="data/manual_grasp/manual_grasps.json"
GRASP_JSON="/home/sjm/franka_grasp_ws/src/franka_grasp_demo/data/graspnet_result/grasp_result.json"

if [ ! -f "$MANUAL_JSON" ]; then
  echo "ERROR: missing $MANUAL_JSON"
  echo "Please run manual_click_grasp_points.py first."
  exit 1
fi

TOTAL=$(python3 - <<'PY'
import json
d=json.load(open("data/manual_grasp/manual_grasps.json"))
print(len(d.get("grasps", [])))
PY
)

if [ "$TOTAL" -le 0 ]; then
  echo "ERROR: no manual grasp points found."
  exit 1
fi

START_INDEX=${1:-0}
END_INDEX=${2:-$((TOTAL - 1))}

echo "Manual grasp sequence:"
echo "  total points : $TOTAL"
echo "  start index  : $START_INDEX"
echo "  end index    : $END_INDEX"
echo

BRIDGE_PID=""

cleanup() {
  if [ -n "$BRIDGE_PID" ]; then
    kill "$BRIDGE_PID" 2>/dev/null || true
    wait "$BRIDGE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for IDX in $(seq "$START_INDEX" "$END_INDEX"); do
  echo "========================================"
  echo "Executing manual grasp index $IDX"
  echo "========================================"

  python3 scripts/select_manual_grasp.py --index "$IDX"

  echo
  echo "Current grasp_result.json:"
  python3 - <<'PY'
import json
d=json.load(open("data/graspnet_result/grasp_result.json"))
print("manual_index:", d.get("manual_index"))
print("position:", d.get("position"))
print("pixel:", d.get("pixel"))
print("z_offset:", d.get("z_offset"))
PY

  echo
  echo "Starting grasp bridge..."
  ros2 run franka_grasp_demo graspnet_bridge_node.py \
    --ros-args \
    -p json_path:="$GRASP_JSON" \
    -p publish_once:=false &
  BRIDGE_PID=$!

  sleep 2

  echo "Starting executor..."
  ros2 launch franka_grasp_demo executor.launch.py

  echo "Executor finished for index $IDX"

  echo "Stopping grasp bridge..."
  kill "$BRIDGE_PID" 2>/dev/null || true
  wait "$BRIDGE_PID" 2>/dev/null || true
  BRIDGE_PID=""

  echo "Waiting 2 seconds before next grasp..."
  sleep 2
done

echo
echo "All selected manual grasps finished."
