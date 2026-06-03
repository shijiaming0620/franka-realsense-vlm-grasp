#!/usr/bin/env python3
import argparse
import base64
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
from openai import OpenAI


def image_to_data_url(path):
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def parse_crop_roi(s, w, h):
    if not s:
        return [0, 0, w, h]
    vals = [int(float(x)) for x in s.split(",")]
    if len(vals) != 4:
        raise RuntimeError("--crop_roi must be x1,y1,x2,y2")
    x1, y1, x2, y2 = vals
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(1, min(w, x2))
    y2 = max(1, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError("Invalid crop_roi")
    return [x1, y1, x2, y2]


def draw_grid(img, step=50):
    out = img.copy()
    h, w = out.shape[:2]
    overlay = out.copy()

    for x in range(0, w, step):
        cv2.line(overlay, (x, 0), (x, h - 1), (0, 255, 255), 1)
        cv2.putText(overlay, str(x), (x + 3, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    for y in range(0, h, step):
        cv2.line(overlay, (0, y), (w - 1, y), (0, 255, 255), 1)
        cv2.putText(overlay, str(y), (3, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return cv2.addWeighted(overlay, 0.35, out, 0.65, 0)


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise RuntimeError("No JSON found in model output:\n" + text)
    return json.loads(m.group(0))


def clamp_point(p, w, h):
    u = int(round(float(p[0])))
    v = int(round(float(p[1])))
    u = max(0, min(w - 1, u))
    v = max(0, min(h - 1, v))
    return [u, v]


def clamp_bbox(b, w, h):
    x1 = max(0, min(w - 1, int(round(float(b[0])))))
    y1 = max(0, min(h - 1, int(round(float(b[1])))))
    x2 = max(0, min(w - 1, int(round(float(b[2])))))
    y2 = max(0, min(h - 1, int(round(float(b[3])))))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def add_offset_point(p, ox, oy):
    return [int(p[0] + ox), int(p[1] + oy)]


def add_offset_bbox(b, ox, oy):
    return [int(b[0] + ox), int(b[1] + oy), int(b[2] + ox), int(b[3] + oy)]


def clean_polygon(points, w, h):
    pts = [clamp_point(p, w, h) for p in points]
    unique = []
    for p in pts:
        if p not in unique:
            unique.append(p)

    if len(unique) < 3:
        return unique

    arr = np.array(unique, dtype=np.float32)
    c = arr.mean(axis=0)
    angles = np.arctan2(arr[:, 1] - c[1], arr[:, 0] - c[0])
    order = np.argsort(angles)
    arr = arr[order]

    return [[int(round(p[0])), int(round(p[1]))] for p in arr]


def polygon_centroid(points):
    pts = np.array(points, dtype=np.float32)

    if len(pts) < 3:
        c = pts.mean(axis=0)
        return [int(round(c[0])), int(round(c[1]))]

    contour = pts.reshape(-1, 1, 2)
    m = cv2.moments(contour)

    if abs(m["m00"]) < 1e-6:
        c = pts.mean(axis=0)
    else:
        c = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]], dtype=float)

    return [int(round(c[0])), int(round(c[1]))]


def direction_from_polygon_short_axis(points):
    pts = np.array(points, dtype=np.float32)

    if len(pts) < 3:
        c = pts.mean(axis=0)
        return [int(round(c[0] + 70)), int(round(c[1]))]

    c = pts.mean(axis=0)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.float32)

    edges = []
    for i in range(4):
        p1 = box[i]
        p2 = box[(i + 1) % 4]
        d = p2 - p1
        length = float(np.linalg.norm(d))
        edges.append((length, d))

    edges.sort(key=lambda x: x[0])
    short_len, short_d = edges[0]
    long_len, _ = edges[-1]

    if long_len < 1e-6 or long_len / max(short_len, 1e-6) < 1.15:
        d = np.array([70.0, 0.0], dtype=float)
    else:
        n = np.linalg.norm(short_d)
        d = np.array([70.0, 0.0], dtype=float) if n < 1e-6 else short_d / n * 70.0

    q = c + d
    return [int(round(q[0])), int(round(q[1]))]


def bbox_area(b):
    return max(1, (b[2] - b[0]) * (b[3] - b[1]))


def ascii_label(text, fallback):
    text = str(text) if text is not None else ""
    keep = []
    for ch in text:
        if ch.isascii() and (ch.isalnum() or ch in ["_", "-", " "]):
            keep.append(ch)
    out = "".join(keep).strip()
    return (out or fallback)[:18]


def process_object(obj, w, h, idx):
    obj["object_id"] = int(obj.get("object_id", idx))
    obj["object_name_en"] = ascii_label(
        obj.get("object_name_en", obj.get("target_description", "")),
        f"object_{idx}"
    )

    if "object_bbox" in obj and len(obj["object_bbox"]) == 4:
        obj["object_bbox"] = clamp_bbox(obj["object_bbox"], w, h)

    if "top_surface_polygon" in obj and len(obj["top_surface_polygon"]) >= 3:
        poly = clean_polygon(obj["top_surface_polygon"], w, h)
        obj["top_surface_polygon"] = poly
        obj["center_pixel"] = clamp_point(polygon_centroid(poly), w, h)
        obj["direction_pixel"] = clamp_point(direction_from_polygon_short_axis(poly), w, h)
    else:
        if "object_bbox" in obj and len(obj["object_bbox"]) == 4:
            x1, y1, x2, y2 = obj["object_bbox"]
            obj["center_pixel"] = [(x1 + x2) // 2, (y1 + y2) // 2]
            obj["direction_pixel"] = [min(w - 1, obj["center_pixel"][0] + 70), obj["center_pixel"][1]]

    confidence = float(obj.get("confidence", 0.0))
    graspability = float(obj.get("graspability", 0.5))
    has_top = 1.0 if "top_surface_polygon" in obj and len(obj["top_surface_polygon"]) >= 3 else 0.4
    area_score = min(1.0, bbox_area(obj.get("object_bbox", [0, 0, 1, 1])) / 8000.0)
    obj["auto_score"] = float(confidence * graspability * has_top * (0.5 + 0.5 * area_score))
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out_json", default="data/manual_grasp/vlm_result.json")
    parser.add_argument("--model", default=os.getenv("VLM_MODEL", "qwen-vl-plus"))
    parser.add_argument("--base_url", default=os.getenv("VLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    parser.add_argument("--api_key_env", default="VLM_API_KEY")
    parser.add_argument("--crop_roi", default="")
    parser.add_argument(
        "--instruction",
        default="识别当前画面中所有适合从上方二指夹取的物体，并自动选择一个最适合抓取的目标。"
    )
    args = parser.parse_args()

    api_key = (
        os.getenv(args.api_key_env)
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError("No API key found. Please export VLM_API_KEY or DASHSCOPE_API_KEY.")

    orig_img = cv2.imread(args.image)
    if orig_img is None:
        raise RuntimeError(f"Cannot read image: {args.image}")

    orig_h, orig_w = orig_img.shape[:2]
    x1, y1, x2, y2 = parse_crop_roi(args.crop_roi, orig_w, orig_h)

    crop_img = orig_img[y1:y2, x1:x2].copy()
    h, w = crop_img.shape[:2]

    out_dir = Path(args.out_json).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_path = out_dir / "vlm_query_grid.png"
    grid_img = draw_grid(crop_img, step=50)
    cv2.imwrite(str(grid_path), grid_img)

    prompt = f"""
你是机器人抓取视觉模块。现在要识别多个可抓取物体，并自动选择一个最适合从上方二指夹爪抓取的目标。

注意：你看到的是工作区域裁剪图，不是完整原图。
裁剪图坐标规则：
- 原点在裁剪图左上角
- u/x 向右增大
- v/y 向下增大
- 裁剪图尺寸：width={w}, height={h}
- 图中黄色网格和数字只是辅助定位

任务：
{args.instruction}

必须输出所有明显可抓取物体，不要只输出一个。
每个物体都要输出：
1. object_id
2. object_name_en：英文短名称，例如 cup、bottle、box、block、cube、tape
3. target_description
4. object_bbox：必须是紧贴整个可见物体的二维框 [x1, y1, x2, y2]
5. top_surface_polygon：可见上表面区域边界点，至少 3 到 6 个点
6. confidence
7. graspability
8. reason

要求：
- object_bbox 要尽量紧，不要包含大量桌面、标定板或背景
- top_surface_polygon 只能框物体上表面，不要框侧壁
- 抓取点必须位于上表面内部中心
- 夹爪方向沿上表面短轴或窄边
- 不要把桌面、标定板、白色胶带边框、阴影当成物体

只输出 JSON，不要 markdown，不要解释。
JSON 格式：
{{
  "objects": [
    {{
      "object_id": 0,
      "object_name_en": "cup",
      "target_description": "物体描述",
      "object_bbox": [x1, y1, x2, y2],
      "top_surface_polygon": [[u1, v1], [u2, v2], [u3, v3], [u4, v4]],
      "confidence": 0.0,
      "graspability": 0.0,
      "reason": "一句话原因"
    }}
  ],
  "selected_object_id": 0,
  "selection_reason": "为什么自动选择这个物体"
}}
"""

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    completion = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(grid_path)}},
                ],
            }
        ],
        temperature=0.05,
    )

    raw_text = completion.choices[0].message.content
    data = extract_json(raw_text)

    objects = data.get("objects", [])
    if not objects:
        raise RuntimeError("VLM returned no objects:\n" + raw_text)

    processed_crop = [process_object(obj, w, h, i) for i, obj in enumerate(objects)]

    processed = []
    for obj in processed_crop:
        obj2 = dict(obj)
        if "object_bbox" in obj2 and len(obj2["object_bbox"]) == 4:
            obj2["object_bbox"] = add_offset_bbox(obj2["object_bbox"], x1, y1)
        if "top_surface_polygon" in obj2:
            obj2["top_surface_polygon"] = [add_offset_point(p, x1, y1) for p in obj2["top_surface_polygon"]]
        if "center_pixel" in obj2:
            obj2["center_pixel"] = add_offset_point(obj2["center_pixel"], x1, y1)
        if "direction_pixel" in obj2:
            obj2["direction_pixel"] = add_offset_point(obj2["direction_pixel"], x1, y1)
        processed.append(obj2)

    best_idx = int(np.argmax([o.get("auto_score", 0.0) for o in processed]))
    best = processed[best_idx]

    data["objects"] = processed
    data["selected_index"] = best_idx
    data["selected_object_id"] = best.get("object_id", best_idx)

    data["target_description"] = best.get("target_description", f"object_{best_idx}")
    data["object_name_en"] = best.get("object_name_en", f"object_{best_idx}")
    data["object_bbox"] = best.get("object_bbox")
    data["top_surface_polygon"] = best.get("top_surface_polygon")
    data["center_pixel"] = best.get("center_pixel")
    data["direction_pixel"] = best.get("direction_pixel")
    data["confidence"] = best.get("confidence", 0.0)
    data["graspability"] = best.get("graspability", 0.0)
    data["auto_score"] = best.get("auto_score", 0.0)

    data["image_width"] = orig_w
    data["image_height"] = orig_h
    data["crop_roi"] = [x1, y1, x2, y2]
    data["model"] = args.model
    data["base_url"] = args.base_url
    data["query_image"] = str(grid_path)

    vis = orig_img.copy()

    # 显示裁剪区域
    cv2.rectangle(vis, (x1, y1), (x2, y2), (80, 80, 80), 2)

    for i, obj in enumerate(processed):
        selected = (i == best_idx)
        bbox_color = (0, 255, 0) if selected else (255, 255, 0)
        poly_color = (255, 0, 255) if selected else (180, 0, 180)

        if "object_bbox" in obj and len(obj["object_bbox"]) == 4:
            bx1, by1, bx2, by2 = obj["object_bbox"]
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), bbox_color, 3 if selected else 2)
            name = obj.get("object_name_en", f"object_{i}")
            label = f"{i} {name} {obj.get('auto_score', 0):.2f}"
            if selected:
                label = "SELECTED " + label
            cv2.putText(vis, label, (bx1, max(25, by1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, bbox_color, 2)

        if "top_surface_polygon" in obj and len(obj["top_surface_polygon"]) >= 3:
            pts = np.array(obj["top_surface_polygon"], dtype=np.int32)
            cv2.polylines(vis, [pts], isClosed=True, color=poly_color, thickness=3 if selected else 2)
            for p in obj["top_surface_polygon"]:
                cv2.circle(vis, tuple(p), 4, poly_color, -1)

    c = data["center_pixel"]
    d = data["direction_pixel"]
    cv2.circle(vis, tuple(c), 9, (0, 0, 255), -1)
    cv2.circle(vis, tuple(d), 6, (0, 0, 255), -1)
    cv2.line(vis, tuple(c), tuple(d), (0, 0, 255), 3)

    preview_path = out_dir / "vlm_result_preview.png"
    cv2.imwrite(str(preview_path), vis)
    data["preview_image"] = str(preview_path)

    with open(args.out_json, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
