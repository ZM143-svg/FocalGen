#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按图像亮度把测试集拆成 normal / lowlight 两个文件夹，拆完直接用原测试命令评测。

用法
----
    python data/split_illumination.py --data-path D:/Dronecrowd/test_data \
        --out-dir D:/Dronecrowd/splits

输出（images / ground_truth 默认硬链接，不额外占磁盘）
-----------------------------------------------------
    D:/Dronecrowd/splits/
    ├── normal/test_data/{images,ground_truth}
    ├── lowlight/test_data/{images,ground_truth}
    └── summary.txt

评测（和平时一样，只换 data_path）
----------------------------------
    python main.py --config-name=dronecrowd \
        data_path=D:/Dronecrowd/splits/lowlight \
        test_only=True debug=True \
        restore_from_ckpt=/path/to/ckpt.ckpt

说明
----
默认按【序列级】平均亮度分组。DroneCrowd 的昼/夜是按序列连续分布的，
逐帧划分会把同一场景的相邻帧拆到两组，引入场景差异的混淆因素。
阈值默认用一维 k-means 自动确定，也可用 --threshold 手动指定。
"""

import argparse
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def gray_mean(path: Path) -> float:
    """图像灰度均值。用 1/8 分辨率读取加速，不影响亮度统计。"""
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()) if img is not None else 0.0


def kmeans_threshold(values: np.ndarray, iters: int = 200) -> float:
    """一维 k-means(k=2)，以最小/最大值为初始簇心（结果确定可复现）。"""
    c_low, c_high = float(values.min()), float(values.max())
    if c_high - c_low < 1e-6:
        return c_low
    for _ in range(iters):
        mid = (c_low + c_high) / 2.0
        low, high = values[values <= mid], values[values > mid]
        new_low = float(low.mean()) if low.size else c_low
        new_high = float(high.mean()) if high.size else c_high
        if abs(new_low - c_low) < 1e-6 and abs(new_high - c_high) < 1e-6:
            break
        c_low, c_high = new_low, new_high
    return (c_low + c_high) / 2.0


def place(src: Path, dst: Path, hardlink: bool) -> None:
    if dst.exists():
        return
    if hardlink:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description='按亮度把测试集拆成 normal / lowlight 两个文件夹')
    ap.add_argument('--data-path', required=True, help='输入目录，需含 images/ 与 ground_truth/')
    ap.add_argument('--out-dir', required=True, help='输出目录')
    ap.add_argument('--threshold', type=float, default=None,
                    help='灰度阈值（越低越暗）；不指定则用一维 k-means 自动确定')
    ap.add_argument('--frame-level', action='store_true', help='按帧分组（默认按序列分组）')
    ap.add_argument('--copy', action='store_true', help='复制文件而非硬链接')
    ap.add_argument('--workers', type=int, default=8, help='读取图像的并发线程数')
    args = ap.parse_args()

    src_root, out_root = Path(args.data_path), Path(args.out_dir)
    src_images, src_gt = src_root / 'images', src_root / 'ground_truth'

    if not src_images.is_dir():
        sys.exit(f'[错误] 找不到图像目录: {src_images}')
    # 数据集代码用字符串替换定位标注（images->ground_truth, img->GT_img），路径含 img 会出错
    if 'img' in str(out_root).lower():
        sys.exit(f'[错误] 输出路径不能包含 "img"（会与标注路径替换冲突）: {out_root}')

    img_paths = sorted(src_images.glob('*.jpg'))
    if not img_paths:
        sys.exit(f'[错误] {src_images} 下没有 .jpg 文件')
    print(f'共 {len(img_paths)} 张图像，计算亮度 ...')

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        means = list(pool.map(gray_mean, img_paths))

    def seq_of(p: Path) -> str:
        """序列号：img 后除末 3 位帧号以外的部分。"""
        d = p.stem[3:] if p.stem.startswith('img') else p.stem
        return d[:-3] if len(d) > 3 else d

    seq_avg = {}
    for p, m in zip(img_paths, means):
        seq_avg.setdefault(seq_of(p), []).append(m)
    seq_avg = {s: float(np.mean(v)) for s, v in seq_avg.items()}

    values = np.array(means if args.frame_level else list(seq_avg.values()), dtype=np.float64)
    thr = args.threshold if args.threshold is not None else kmeans_threshold(values)

    if args.frame_level:
        low_sel = [p for p, m in zip(img_paths, means) if m <= thr]
    else:
        low_seqs = {s for s, v in seq_avg.items() if v <= thr}
        low_sel = [p for p in img_paths if seq_of(p) in low_seqs]
    low_set = set(low_sel)
    groups = {'normal': [p for p in img_paths if p not in low_set], 'lowlight': low_sel}

    lines = ['光照分组结果', '=' * 56,
             f'输入     : {src_root}',
             f'图像总数 : {len(img_paths)}',
             f'分组粒度 : {"逐帧" if args.frame_level else "序列级（推荐）"}',
             f'灰度阈值 : {thr:.2f}'
             + ('' if args.threshold is not None else '（k-means 自动确定）'),
             '']
    mean_of = dict(zip(img_paths, means))

    for name in ('normal', 'lowlight'):
        items = groups[name]
        if not items:
            print(f'[警告] {name} 组为空'); continue

        dst_images = out_root / name / 'test_data' / 'images'
        dst_gt = out_root / name / 'test_data' / 'ground_truth'
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_gt.mkdir(parents=True, exist_ok=True)

        missing = 0
        for p in items:
            place(p, dst_images / p.name, not args.copy)
            gt = src_gt / p.name.replace('.jpg', '.mat').replace('img', 'GT_img')
            if gt.is_file():
                place(gt, dst_gt / gt.name, not args.copy)
            else:
                missing += 1

        seqs = sorted({seq_of(p) for p in items})
        msg = (f'{name:<9}: {len(items):>5} 帧 / {len(seqs):>2} 序列'
               f'  平均亮度 {np.mean([mean_of[p] for p in items]):>6.2f}')
        if missing:
            msg += f'  [警告] {missing} 个标注缺失'
        print(msg)
        lines += [msg, f'  序列: {" ".join(seqs)}', f'  目录: {out_root / name}']

    summary = '\n'.join(lines)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / 'summary.txt').write_text(summary, encoding='utf-8')
    print('\n' + summary)
    print(f'\n评测时把 data_path 指向 {out_root}/normal 或 {out_root}/lowlight 即可。')


if __name__ == '__main__':
    main()
