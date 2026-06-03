#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="/home/sjm/franka_grasp_ws/src/franka_grasp_demo/data/graspnet_result/grasp_result_raw.json")
    parser.add_argument("--out", default="/home/sjm/franka_grasp_ws/src/franka_grasp_demo/data/graspnet_result/grasp_result.json")
    parser.add_argument("--set_z", type=float, default=None, help="Manually set execution z. If omitted, use raw z.")
    parser.add_argument("--z_offset", type=float, default=0.0, help="Add offset to raw z when --set_z is omitted.")
    args = parser.parse_args()

    raw_path = Path(args.raw)
    out_path = Path(args.out)

    data = json.load(open(raw_path, "r"))

    raw_pos = data["position"].copy()
    raw_ori = data["orientation"].copy()

    if args.set_z is not None:
        data["position"][2] = args.set_z
        data["note"] = f"EXECUTION GRASP: no compensation, z manually set to {args.set_z:.3f}."
    else:
        data["position"][2] = raw_pos[2] + args.z_offset
        data["note"] = f"EXECUTION GRASP: no compensation, z = raw_z + {args.z_offset:.3f}."

    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(out_path, "w"), indent=2)

    print("raw position:", raw_pos)
    print("raw orientation:", raw_ori)
    print("final position:", data["position"])
    print("final orientation:", data["orientation"])
    print("saved:", out_path)


if __name__ == "__main__":
    main()
