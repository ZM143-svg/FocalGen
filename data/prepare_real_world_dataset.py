"""把 `Real-world-dataset.zip`（或解压后的目录）转换成 `merged_yolo_pixel` 布局。

zip 内是**按序列划分的原始标注**：

    dataset1/
    ├── images/frame_00001.jpg            # 1920x1080
    ├── yolo_labels/frame_00001.txt       # YOLO: "class cx cy w h"（归一化）
    ├── yolo_labels/classes.txt           # "Person head"
    ├── _annotations.coco.json            # COCO bbox 导出
    ├── frame_00001.json                  # 单帧 COCO（可选）
    ├── yolo.py / fenjson.py              # 作者原始的转换脚本
    └── ...
    dataset2/ ... dataset12/

而 `dataset: merged_yolo_pixel`（`configs/dronecrowd.yaml`）需要的是**扁平、同名配对**的布局：

    <out>/
    ├── images/dataset1_frame_00001.jpg
    └── ground_truth/dataset1_frame_00001.txt     # 每行 "x y"（框中心像素坐标）

用法::

    # 直接从 zip 转换（推荐）
    python data/prepare_real_world_dataset.py --zip data/Real-world-dataset.zip --out D:/Real-world-dataset

    # 推荐：标注直接用仓库自带的 data/ground_truth（与论文完全一致），只从 zip 提取/改名图片
    python data/prepare_real_world_dataset.py --zip data/Real-world-dataset.zip \\
        --out D:/Real-world-dataset --gt-dir data/ground_truth

    # 或者用已经解压好的目录
    python data/prepare_real_world_dataset.py --src D:/1 --out D:/Real-world-dataset

    # 只要标注不要图片（用于和仓库自带的 data/ground_truth 对比）
    python data/prepare_real_world_dataset.py --zip data/Real-world-dataset.zip --out ./tmp --no-images

标注生成规则：点坐标 = `floor(框中心 * 图像宽高)`，已对 **128 帧 / 5530 个点** 逐一核对：
128 个 txt 中有 98 个与仓库 `data/ground_truth/` **逐字节相同**，其余 30 个各有 1 个点差 1px
（YOLO 归一化坐标只有 6 位小数，边界处取整会差 1）。**要精确复现论文数字请加 `--gt-dir data/ground_truth`。**
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import zipfile

IMG_RE = re.compile(r"^dataset(\d+)/images/(frame_\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def load_image_size(img_rel: str, zip_file: zipfile.ZipFile | None, src_dir: str | None) -> tuple[int, int]:
    """从该序列的 COCO json 读取图像宽高，失败则用 cv2 / PIL 直接从图片读取。"""
    seq = img_rel.split("/")[0]
    coco_rel = f"{seq}/_annotations.coco.json"

    raw = None
    if zip_file is not None and coco_rel in zip_file.namelist():
        raw = zip_file.read(coco_rel).decode("utf-8")
    elif src_dir is not None:
        coco_path = os.path.join(src_dir, seq, "_annotations.coco.json")
        if os.path.exists(coco_path):
            with open(coco_path, "r", encoding="utf-8") as f:
                raw = f.read()

    base = os.path.basename(img_rel)
    if raw is not None:
        try:
            coco = json.loads(raw)
            for info in coco.get("images", []):
                if info.get("file_name") == base:
                    return int(info["width"]), int(info["height"])
        except (ValueError, KeyError):
            pass

    # 回退：直接读图片
    if zip_file is not None:
        import io

        data = zip_file.read(img_rel)
        try:
            import cv2
            import numpy as np

            arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if arr is not None:
                return arr.shape[1], arr.shape[0]
        except ImportError:
            try:
                from PIL import Image

                with Image.open(io.BytesIO(data)) as im:
                    return im.width, im.height
            except ImportError:
                pass
    elif src_dir is not None:
        try:
            import cv2

            arr = cv2.imread(os.path.join(src_dir, img_rel), cv2.IMREAD_COLOR)
            if arr is not None:
                return arr.shape[1], arr.shape[0]
        except ImportError:
            try:
                from PIL import Image

                with Image.open(os.path.join(src_dir, img_rel)) as im:
                    return im.width, im.height
            except ImportError:
                pass

    raise RuntimeError(f"无法获取图像尺寸：{img_rel}（请安装 opencv-python 或 pillow）")


def parse_yolo(txt: str) -> list[tuple[float, float]]:
    """YOLO 标注 -> 归一化框中心列表 [(cx, cy), ...]。"""
    centers = []
    for line in txt.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            centers.append((float(parts[1]), float(parts[2])))
        except ValueError:
            continue
    return centers


def link_or_copy(src: str, dst: str) -> str:
    """优先硬链接（同盘、秒完成），失败则复制。"""
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main() -> int:
    ap = argparse.ArgumentParser(description="Real-world-dataset -> merged_yolo_pixel 转换")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--zip", help="Real-world-dataset.zip 路径")
    g.add_argument("--src", help="已解压的数据集根目录（内含 dataset1/ ... dataset12/）")
    ap.add_argument("--out", required=True, help="输出根目录，将生成 images/ 与 ground_truth/")
    ap.add_argument("--no-images", action="store_true", help="只生成 ground_truth/，不处理图片")
    ap.add_argument("--gt-dir", default=None,
                    help="直接拷贝该目录下的 ground_truth（如仓库的 data/ground_truth），"
                         "而不是从 yolo 标签换算，用于精确复现论文结果")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    args = ap.parse_args()

    out_images = os.path.join(args.out, "images")
    out_gt = os.path.join(args.out, "ground_truth")
    if not args.dry_run:
        os.makedirs(out_gt, exist_ok=True)
        if not args.no_images:
            os.makedirs(out_images, exist_ok=True)

    zip_file = zipfile.ZipFile(args.zip) if args.zip else None
    import tempfile

    tmp_dir = None
    try:
        if zip_file is not None:
            names = zip_file.namelist()
        else:
            names = []
            for root, _dirs, files in os.walk(args.src):
                for fn in files:
                    p = os.path.relpath(os.path.join(root, fn), args.src).replace("\\", "/")
                    names.append(p)

        images = sorted(n for n in names if IMG_RE.match(n))
        if not images:
            print("未找到任何 datasetX/images/frame_YYYYY.jpg，请检查输入路径。", file=sys.stderr)
            return 1

        n_pts = 0
        n_missing = 0
        n_missing_gt = 0
        for img_rel in images:
            m = IMG_RE.match(img_rel)
            seq, frame, ext = m.group(1), m.group(2), m.group(3)
            label_rel = f"dataset{seq}/yolo_labels/{frame}.txt"

            label_txt = None
            if zip_file is not None and label_rel in names:
                label_txt = zip_file.read(label_rel).decode("utf-8")
            elif zip_file is None and os.path.exists(os.path.join(args.src, label_rel)):
                with open(os.path.join(args.src, label_rel), "r", encoding="utf-8") as f:
                    label_txt = f.read()

            stem = f"dataset{seq}_{frame}"
            if label_txt is None:
                n_missing += 1
                print(f"  [warn] 缺少标注，跳过：{label_rel}")
                continue

            width, height = load_image_size(img_rel, zip_file, args.src)
            centers = parse_yolo(label_txt)
            n_pts += len(centers)

            if args.dry_run:
                continue

            # ground_truth：优先用 --gt-dir 指定的现成标注，否则由框中心换算
            gt_path = os.path.join(out_gt, stem + ".txt")
            src_gt = os.path.join(args.gt_dir, stem + ".txt") if args.gt_dir else None
            if src_gt and os.path.exists(src_gt):
                shutil.copyfile(src_gt, gt_path)
            else:
                with open(gt_path, "w", encoding="utf-8") as f:
                    for cx, cy in centers:
                        f.write(f"{int(cx * width)} {int(cy * height)}\n")
                if src_gt:
                    n_missing_gt += 1

            # images：改名成 "<序列>_<帧>.jpg" 与标注同名
            if not args.no_images:
                dst = os.path.join(out_images, stem + "." + ext.lower())
                if zip_file is not None:
                    if tmp_dir is None:
                        tmp_dir = tempfile.mkdtemp(prefix="rwd_")
                    tmp_img = os.path.join(tmp_dir, stem + "." + ext.lower())
                    with zip_file.open(img_rel) as src_f, open(tmp_img, "wb") as dst_f:
                        shutil.copyfileobj(src_f, dst_f)
                    link_or_copy(tmp_img, dst)
                else:
                    link_or_copy(os.path.join(args.src, img_rel), dst)

        print(f"帧数：{len(images) - n_missing}（缺标注 {n_missing}）  点数：{n_pts}")
        if args.gt_dir:
            print(f"标注来源：{args.gt_dir}（其中 {n_missing_gt} 帧在源目录里没找到，已由 yolo 标签换算）")
        print(f"images      -> {out_images}")
        print(f"ground_truth-> {out_gt}")
        if args.dry_run:
            print("（--dry-run：未写入任何文件）")
        return 0
    finally:
        if zip_file is not None:
            zip_file.close()
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
