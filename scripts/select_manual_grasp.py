import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, required=True)
parser.add_argument("--manual_json", default="data/manual_grasp/manual_grasps.json")
parser.add_argument("--out", default="data/graspnet_result/grasp_result.json")
args = parser.parse_args()

data = json.load(open(args.manual_json))
grasps = data["grasps"]

if args.index < 0 or args.index >= len(grasps):
    raise SystemExit(f"index {args.index} out of range, total={len(grasps)}")

g = grasps[args.index]

out = {
    "frame_id": "base",
    "position": g["base_position"],
    "orientation": [0.0, 0.0, 0.0, 1.0],
    "width": 0.075,
    "score": 1.0,
    "source": "manual_click",
    "manual_index": args.index,
    "pixel": g["pixel"],
    "camera_position": g["camera_position"],
    "base_position_raw": g["base_position_raw"],
    "z_offset": g["z_offset"],
    "note": "EXECUTION GRASP: manual click point. Orientation ignored by manual-click executor.",
}

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(args.out, "w"), indent=2)

print("selected manual index:", args.index)
print("pixel:", g["pixel"])
print("base raw:", g["base_position_raw"])
print("base final:", g["base_position"])
print("saved:", args.out)
