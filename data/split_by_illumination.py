#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按光照条件（亮度）把测试集划分为 Normal illumination / Low illumination 两组。

分组粒度说明
------------
DroneCrowd 的图像命名为 img + 3 位序列号 + 3 位帧号（如 img011001 = 序列 011 第 001 帧），
同一序列内场景、背景、密度高度相关。因此默认按【序列级平均亮度】分组（scene-level），
而不是逐帧分组，避免同一场景的相邻帧被拆到不同组、引入场景内容差异的混淆因素。

用法
----
    # 默认：序列级 + 1D k-means 自动定阈值
    python data/split_by_illumination.py --data-path D:/Dronecrowd/test_data

    # 指定输出目录与分组方式
    python data/split_by_illumination.py --data-path D:/Dronecrowd/test_data \
        --out-dir data/splits --mode percentile --low-percentile 35

    # 逐帧分组（不推荐，仅作对照）
    python data/split_by_illumination.py --data-path D:/Dronecrowd/test_data --frame-level

输出
----
    <out-dir>/frame_stats.csv        每帧亮度指标
    <out-dir>/sequence_stats.csv     每序列亮度指标
    <out-dir>/normal.txt             Normal 组图像绝对路径（每行一个）
    <out-dir>/lowlight.txt           Low illumination 组图像绝对路径
    <out-dir>/illumination_hist.png  亮度分布直方图（需要 matplotlib，可选）
    <out-dir>/summary.txt            分组汇总（阈值、数量、序列列表）
"""

import argparse
import csv
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# 亮度计算
# --------------------------------------------------------------------------- #
def parse_sequence(stem: str):
    """从文件名 stem（如 'img011001'）解析 (序列号, 帧号)。

    兼容 6 位（3 位序列 + 3 位帧）与其它位数：取末 3 位为帧号，其余为序列号。
    解析失败时返回 ('unknown', stem)。
    """
    m = re.match(r'^img(\d+)$', stem)
    if not m:
        return 'unknown', stem
    digits = m.group(1)
    if len(digits) > 3:
        return digits[:-3], digits[-3:]
    return digits, digits


def image_brightness(path: Path):
    """读取图像并计算亮度指标。

    使用 IMREAD_REDUCED_COLOR_8（1/8 分辨率）加速，亮度统计不受影响。
    返回 (mean, p10, p50, p90, low_frac, ok)
      - mean     : 灰度均值 (0~255)
      - p10/p50/p90 : 灰度分位数
      - low_frac : 暗像素占比（灰度 < 50 的比例），用于刻画"整体偏暗"还是"局部欠曝"
    """
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
    if img is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0, False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flat = gray.reshape(-1).astype(np.float32)
    p10, p50, p90 = np.percentile(flat, [10, 50, 90])
    return float(flat.mean()), float(p10), float(p50), float(p90), \
        float((flat < 50).mean()), True


# --------------------------------------------------------------------------- #
# 阈值确定
# --------------------------------------------------------------------------- #
def _otsu_1d(values: np.ndarray, nbins: int = 256):
    """在取值范围内做 Otsu 阈值分割，返回阈值。"""
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-6:
        return lo
    hist, edges = np.histogram(values, bins=nbins, range=(lo, hi))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0

    total = hist.sum()
    w0 = np.cumsum(hist) / total
    w1 = 1.0 - w0
    csum = np.cumsum(hist * centers)
    mean_total = csum[-1]
    mu0 = np.divide(csum, np.cumsum(hist), out=np.zeros_like(csum),
                    where=np.cumsum(hist) > 0)
    mu1 = np.divide(mean_total - csum, total - np.cumsum(hist),
                    out=np.zeros_like(csum), where=(total - np.cumsum(hist)) > 0)

    var_between = w0 * w1 * (mu0 - mu1) ** 2
    var_between = np.nan_to_num(var_between)
    return float(centers[int(np.argmax(var_between))])


def _kmeans_1d(values: np.ndarray, iters: int = 200):
    """一维 k-means (k=2)，返回 (阈值, 低亮簇中心, 高亮簇中心)。"""
    c_low, c_high = float(values.min()), float(values.max())
    if abs(c_high - c_low) < 1e-6:
        return c_low, c_low, c_high
    for _ in range(iters):
        split = (c_low + c_high) / 2.0
        low = values[values <= split]
        high = values[values > split]
        new_low = float(low.mean()) if low.size else c_low
        new_high = float(high.mean()) if high.size else c_high
        if abs(new_low - c_low) < 1e-6 and abs(new_high - c_high) < 1e-6:
            break
        c_low, c_high = new_low, new_high
    return (c_low + c_high) / 2.0, c_low, c_high


def decide_threshold(values: np.ndarray, mode: str, low_percentile: float, fixed: float):
    """返回 (阈值, 说明字符串)。"""
    if mode == 'kmeans':
        thr, c_low, c_high = _kmeans_1d(values)
        return thr, f'1D k-means 自动阈值（簇中心 {c_low:.2f} / {c_high:.2f}）'
    if mode == 'otsu':
        thr = _otsu_1d(values)
        return thr, 'Otsu 自动阈值'
    if mode == 'percentile':
        thr = float(np.percentile(values, low_percentile))
        return thr, f'分位数阈值（最低 {low_percentile:g}% 判为 Low）'
    if mode == 'fixed':
        return float(fixed), f'固定阈值 {fixed:g}'
    raise ValueError(f'未知 mode: {mode}')


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description='按光照亮度划分测试集')
    ap.add_argument('--data-path', required=True,
                    help='数据集根目录，需包含 images/ 与 ground_truth/')
    ap.add_argument('--out-dir', default='data/splits',
                    help='输出目录（默认 data/splits）')
    ap.add_argument('--mode', default='kmeans',
                    choices=['kmeans', 'otsu', 'percentile', 'fixed'],
                    help='阈值确定方式（默认 kmeans）')
    ap.add_argument('--low-percentile', type=float, default=35.0,
                    help='mode=percentile 时，最低亮度百分之多少判为 Low')
    ap.add_argument('--threshold', type=float, default=60.0,
                    help='mode=fixed 时使用的亮度阈值')
    ap.add_argument('--frame-level', action='store_true',
                    help='按帧分组（默认按序列分组，更严谨）')
    ap.add_argument('--workers', type=int, default=8,
                    help='读取图像的并发线程数')
    args = ap.parse_args()

    data_path = Path(args.data_path)
    images_dir = data_path / 'images'
    gt_dir = data_path / 'ground_truth'

    if not images_dir.is_dir():
        sys.exit(f'[错误] 找不到图像目录: {images_dir}')

    img_paths = sorted(images_dir.glob('*.jpg'))
    if not img_paths:
        sys.exit(f'[错误] {images_dir} 下没有 .jpg 文件')
    print(f'共发现 {len(img_paths)} 张图像，开始计算亮度 ...')

    # ---- 逐帧计算亮度 ----
    def work(p):
        mean, p10, p50, p90, low_frac, ok = image_brightness(p)
        seq, frame = parse_sequence(p.stem)
        return {
            'path': p, 'name': p.name, 'sequence': seq, 'frame': frame,
            'mean': mean, 'p10': p10, 'p50': p50, 'p90': p90,
            'low_frac': low_frac, 'ok': ok,
        }

    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(work, img_paths))

    bad = [r for r in rows if not r['ok']]
    if bad:
        print(f'[警告] {len(bad)} 张图像读取失败，已按亮度 0 处理。示例: {bad[0]["name"]}')

    # ---- 序列聚合 ----
    seq_groups = {}
    for r in rows:
        seq_groups.setdefault(r['sequence'], []).append(r)

    seq_rows = []
    for seq, items in sorted(seq_groups.items()):
        means = np.array([it['mean'] for it in items], dtype=np.float64)
        seq_rows.append({
            'sequence': seq,
            'n_frames': len(items),
            'mean': float(means.mean()),
            'std': float(means.std()),
            'min': float(means.min()),
            'max': float(means.max()),
        })

    # ---- 决定阈值与分组 ----
    if args.frame_level:
        values = np.array([r['mean'] for r in rows], dtype=np.float64)
        thr, thr_desc = decide_threshold(values, args.mode, args.low_percentile, args.threshold)
        normal = [r for r in rows if r['mean'] > thr]
        low = [r for r in rows if r['mean'] <= thr]
        group_desc = '逐帧分组（frame-level）'
    else:
        values = np.array([s['mean'] for s in seq_rows], dtype=np.float64)
        thr, thr_desc = decide_threshold(values, args.mode, args.low_percentile, args.threshold)
        low_seqs = {s['sequence'] for s in seq_rows if s['mean'] <= thr}
        normal = [r for r in rows if r['sequence'] not in low_seqs]
        low = [r for r in rows if r['sequence'] in low_seqs]
        group_desc = '序列级分组（scene-level，推荐）'

    # ---- 写出结果 ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / 'frame_stats.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['name', 'sequence', 'frame', 'mean',
                                          'p10', 'p50', 'p90', 'low_frac', 'path'])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    with (out_dir / 'sequence_stats.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['sequence', 'n_frames', 'mean', 'std', 'min', 'max'])
        w.writeheader()
        w.writerows(seq_rows)

    def write_list(path: Path, items):
        with path.open('w', encoding='utf-8') as f:
            for it in items:
                f.write(it['path'].as_posix() + '\n')

    write_list(out_dir / 'normal.txt', normal)
    write_list(out_dir / 'lowlight.txt', low)

    # ---- 汇总 ----
    normal_means = np.array([r['mean'] for r in normal]) if normal else np.array([0.0])
    low_means = np.array([r['mean'] for r in low]) if low else np.array([0.0])
    low_seq_sorted = sorted({r['sequence'] for r in low})
    normal_seq_sorted = sorted({r['sequence'] for r in normal})

    lines = [
        '光照分组汇总',
        '=' * 60,
        f'数据根目录 : {data_path}',
        f'图像总数   : {len(rows)}',
        f'分组粒度   : {group_desc}',
        f'阈值方式   : {args.mode} -> {thr_desc}',
        f'亮度阈值   : {thr:.3f}  (越低越暗)',
        '',
        f'Normal illumination : {len(normal)} 帧 / {len(normal_seq_sorted)} 序列'
        f'  平均亮度 {normal_means.mean():.2f}',
        f'Low  illumination   : {len(low)} 帧 / {len(low_seq_sorted)} 序列'
        f'  平均亮度 {low_means.mean():.2f}',
        '',
        f'Low 组序列号  : {", ".join(low_seq_sorted) if low_seq_sorted else "(无)"}',
        f'Normal 组序列号: {", ".join(normal_seq_sorted) if normal_seq_sorted else "(无)"}',
        '',
        '输出文件:',
        f'  {out_dir / "frame_stats.csv"}',
        f'  {out_dir / "sequence_stats.csv"}',
        f'  {out_dir / "normal.txt"}',
        f'  {out_dir / "lowlight.txt"}',
    ]
    summary = '\n'.join(lines)
    (out_dir / 'summary.txt').write_text(summary, encoding='utf-8')
    print()
    print(summary)

    # ---- 可选：亮度分布直方图 ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].hist([r['mean'] for r in rows], bins=80, color='#94a3b8')
        axes[0].axvline(thr, color='#ef4444', linestyle='--', label=f'thr={thr:.1f}')
        axes[0].set_title('Frame-level brightness')
        axes[0].set_xlabel('mean gray value')
        axes[0].set_ylabel('#frames')
        axes[0].legend()

        if not args.frame_level:
            axes[1].hist([s['mean'] for s in seq_rows], bins=40, color='#60a5fa')
            axes[1].axvline(thr, color='#ef4444', linestyle='--', label=f'thr={thr:.1f}')
            axes[1].set_title('Sequence-level brightness')
            axes[1].set_xlabel('mean gray value (per sequence)')
            axes[1].set_ylabel('#sequences')
            axes[1].legend()

        fig.tight_layout()
        fig.savefig(out_dir / 'illumination_hist.png', dpi=200)
        print(f'\n已保存亮度分布图 -> {out_dir / "illumination_hist.png"}')
    except Exception as e:  # matplotlib 未安装等情况，不影响主流程
        print(f'[提示] 未生成直方图（{e}）')


if __name__ == '__main__':
    main()
