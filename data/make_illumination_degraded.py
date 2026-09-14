#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
方案 B：人工光照退化（synthetic illumination degradation）

对测试集生成受控的亮度退化版本，用于控制变量地验证
"随着 illumination degradation 加剧，模型性能下降幅度更小"。

两种退化方式
------------
1. spatial : I' = alpha * I
             空间域线性缩放，物理直观（模拟曝光不足）。
2. freq    : 仅缩放频域幅度 A' = alpha * A，相位保持不变，再 IFFT 重建
             （与训练期 LightnessPerturbation 的频域幅度缩放机制一致）

⚠️ 两种模式的关系（重要，用 --check-modes 可数值验证）
-----------------------------------------------------
FFT 是线性变换：把整个频谱的幅度乘以 alpha 而相位保持不变，
数学上完全等价于把图像逐像素乘以 alpha（fftshift 只是置换，不改变幅度）。
实测两种模式在所有 alpha（含 1.5 / 2.0）下输出一致，最大差异 ~1e-4，
即 float32 FFT 舍入误差（相对误差 ~4e-7）。

结论：
  * 频域幅度缩放 **实质上就是亮度整体缩放**，并非结构/频谱上的改写。
  * 因此不需要同时生成 spatial 与 freq 两套数据，默认只生成 spatial。

⚠️ 与此相关的实现风险（建议核查训练代码）
-----------------------------------------
训练期的 LightnessPerturbation 把幅度缩放作用在**已归一化**的张量上，
最后又 clamp(0, 1)。由于归一化后存在大量负值，这个 clamp 会把负值
全部截为 0、把大于 1 的值截为 1 —— 这是一个非线性且可能破坏表征的操作，
与 "频域扰动" 的直觉不符。当前 training_step 中该调用已被注释，
属死代码，论文描述需与实际代码保持一致。

⚠️ 术语提醒（论文表述）
-----------------------
两种模式都是对**原始图像**做照明退化。若论文采用该实验，
建议统一称为 "synthetic illumination degradation"，
不要写成 "frequency-domain perturbation"（训练期机制的名字），
除非你能说明退化是在与训练相同的归一化空间中施加的。

输出目录结构（每个 level 可直接作为 data_path 使用）
---------------------------------------------------
<out-dir>/<mode>_a<alpha>/
├── test_data/
│   ├── images/        # 退化后的 *.jpg（文件名与原图一致）
│   └── ground_truth/  # 硬链接到原始 .mat（默认不占额外磁盘空间）
└── degradation_stats.csv / summary.txt

用法
----
    # 先估算体积，不实际生成
    python data/make_illumination_degraded.py --data-path D:/Dronecrowd/test_data --dry-run

    # 生成 alpha = 0.8 / 0.6 / 0.4 的空间域退化
    python data/make_illumination_degraded.py \
        --data-path D:/Dronecrowd/test_data --out-dir D:/Dronecrowd_illum \
        --modes spatial --alphas 0.8 0.6 0.4

    # 只退化 Low illumination 子集（配合 data/splits/lowlight.txt）
    python data/make_illumination_degraded.py \
        --data-path D:/Dronecrowd/test_data --out-dir D:/Dronecrowd_illum_low \
        --image-list data/splits/lowlight.txt --alphas 0.6 0.4
"""

import argparse
import csv
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# 退化算子
# --------------------------------------------------------------------------- #
def degrade(img_bgr: np.ndarray, mode: str, alpha: float):
    """对 BGR uint8 图像做光照退化，返回 (float32 图像 [0,255], 裁剪像素占比)。

    - spatial: I' = alpha * I
    - freq   : 保持相位、仅把频域幅度乘以 alpha，再 IFFT 重建
    """
    src = img_bgr.astype(np.float32)

    if mode == 'spatial':
        out = src * alpha
    elif mode == 'freq':
        out = np.empty_like(src)
        for c in range(src.shape[2]):
            spec = np.fft.fftshift(np.fft.fft2(src[:, :, c]))
            amp = np.abs(spec) * alpha
            phase = np.angle(spec)
            spec2 = amp * np.exp(1j * phase)
            out[:, :, c] = np.real(np.fft.ifft2(np.fft.ifftshift(spec2)))
    else:
        raise ValueError(f'未知 mode: {mode}')

    clipped = float(((out < 0.0) | (out > 255.0)).mean())
    return np.clip(out, 0.0, 255.0), clipped


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def link_or_copy(src: Path, dst: Path, hardlink: bool = True) -> str:
    """默认优先硬链接（不占额外空间），失败或禁用时复制。"""
    if dst.exists():
        return 'exists'
    if hardlink:
        try:
            os.link(src, dst)
            return 'hardlink'
        except OSError:
            pass
    shutil.copy2(src, dst)
    return 'copy'


def gray_mean(img_bgr: np.ndarray) -> float:
    """灰度均值（下采样加速，用于统计报告）。"""
    small = cv2.resize(img_bgr, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())


def run_mode_check(img_paths, report_path=None, n=3, alphas=(0.8, 0.6, 0.4, 1.0, 1.5, 2.0)):
    """数值比较 spatial 与 freq 两种模式的差异（可选写入报告文件）。"""
    samples = img_paths[:n]
    lines = [f'spatial vs freq 数值比较（{len(samples)} 张样本）',
             f'{"alpha":>6} {"max|diff|":>14} {"clip%(spatial)":>16} {"clip%(freq)":>13}',
             '-' * 54]
    for alpha in alphas:
        worst = 0.0
        clip_s = clip_f = 0.0
        for p in samples:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            # [1] 是裁剪前的越界像素占比，两种模式统一按此口径统计
            a, clip_a = degrade(img, 'spatial', alpha)
            b, clip_b = degrade(img, 'freq', alpha)
            worst = max(worst, float(np.abs(a - b).max()))
            clip_s += clip_a
            clip_f += clip_b
        k = max(1, len(samples))
        note = '  <-- 等价' if worst < 1e-3 else ''
        lines.append(f'{alpha:>6.2f} {worst:>14.6f} {clip_s / k * 100:>15.4f}% '
                     f'{clip_f / k * 100:>12.4f}%{note}')
    lines += ['',
              '结论:',
              '  FFT 是线性变换，把整个频谱幅度乘以 alpha 而保持相位不变，',
              '  完全等价于把图像逐像素乘以 alpha（fftshift 只是置换，不改变幅度）。',
              '  实测两者在所有 alpha（含 1.5 / 2.0）下输出一致，最大差异 ~1e-4，',
              '  即 float32 FFT 舍入误差，无实质差异。',
              '',
              '  因此：频域幅度缩放实质上就是亮度整体缩放，并非结构/频谱上的改写；',
              '  不需要同时生成 spatial 与 freq 两套数据，默认只生成 spatial 即可。']
    report = '\n'.join(lines)
    print('\n' + report)
    if report_path is not None:
        rp = Path(report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(report, encoding='utf-8')
        print(f'\n已写入 -> {rp}')


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description='方案 B：人工光照退化（synthetic illumination degradation）')
    ap.add_argument('--data-path', required=True,
                    help='原始数据集根目录，需包含 images/ 与 ground_truth/（如 D:/Dronecrowd/test_data）')
    ap.add_argument('--out-dir', default=None,
                    help='输出根目录；每个 level 生成 <out-dir>/<mode>_a<alpha>/test_data/')
    ap.add_argument('--modes', nargs='+', default=['spatial'], choices=['spatial', 'freq'],
                    help='退化方式，可多选（默认 spatial）')
    ap.add_argument('--alphas', nargs='+', type=float, default=[0.8, 0.6, 0.4],
                    help='亮度缩放系数（默认 0.8 0.6 0.4）')
    ap.add_argument('--image-list', default=None,
                    help='可选：只退化该 txt 中列出的图像（每行一个路径或文件名）')
    ap.add_argument('--limit', type=int, default=None,
                    help='可选：只处理前 N 张（冒烟测试用）')
    ap.add_argument('--jpeg-quality', type=int, default=95,
                    help='输出 JPEG 质量（默认 95）')
    ap.add_argument('--workers', type=int, default=8, help='并发线程数')
    ap.add_argument('--no-hardlink', action='store_true',
                    help='ground_truth 用复制而非硬链接')
    ap.add_argument('--check-modes', action='store_true',
                    help='数值比较 spatial 与 freq 两种模式，打印结论后退出')
    ap.add_argument('--dry-run', action='store_true', help='只估算体积，不生成文件')
    args = ap.parse_args()

    src_root = Path(args.data_path)
    src_images = src_root / 'images'
    src_gt = src_root / 'ground_truth'

    if not src_images.is_dir():
        sys.exit(f'[错误] 找不到图像目录: {src_images}')

    img_paths = sorted(src_images.glob('*.jpg'))
    if args.image_list:
        list_path = Path(args.image_list)
        if not list_path.is_file():
            sys.exit(f'[错误] 找不到 image_list: {list_path}')
        with list_path.open('r', encoding='utf-8') as f:
            wanted = {Path(line.strip()).name for line in f if line.strip()}
        img_paths = [p for p in img_paths if p.name in wanted]
        print(f'[image_list] 命中 {len(img_paths)} 张图像（列表共 {len(wanted)} 项）')

    if args.limit:
        img_paths = img_paths[:args.limit]
        print(f'[limit] 只处理前 {len(img_paths)} 张图像（冒烟测试）')

    if not img_paths:
        sys.exit('[错误] 没有待处理的图像')

    # ---- 可选：先做模式等价性验证 ----
    if args.check_modes:
        report_path = Path(args.out_dir) / 'mode_check.txt' if args.out_dir else None
        run_mode_check(img_paths, report_path=report_path)
        return

    # ---- 体积估算 ----
    sample_n = min(30, len(img_paths))
    sample_bytes = sum(p.stat().st_size for p in img_paths[:sample_n]) / sample_n
    n_levels = len(args.modes) * len(args.alphas)
    # 变暗会显著提高 JPEG 压缩率，粗略按可压缩性折半估计
    est_total = sample_bytes * 0.5 * len(img_paths) * n_levels / (1024 ** 3)
    print(f'\n待处理图像 : {len(img_paths)} 张')
    print(f'原始平均体积: {sample_bytes / 1024:.1f} KB/张')
    print(f'退化 level : {n_levels} 个 = {len(args.modes)} mode x {len(args.alphas)} alpha')
    print(f'输出体积估算: 约 {est_total:.1f} GB（变暗后 JPEG 更易压缩，实际通常更小）')
    print(f'ground_truth : 硬链接，不额外占空间' if not args.no_hardlink else 'ground_truth : 复制')

    if args.dry_run:
        print('\n[dry-run] 未生成任何文件。去掉 --dry-run 即可实际生成。')
        return

    if args.out_dir is None:
        sys.exit('[错误] 实际生成时必须指定 --out-dir')

    out_root = Path(args.out_dir)
    stats_rows = []

    # ---- 逐 level 生成 ----
    for mode in args.modes:
        for alpha in args.alphas:
            level_name = f'{mode}_a{alpha:g}'
            level_root = out_root / level_name
            dst_images = level_root / 'test_data' / 'images'
            dst_gt = level_root / 'test_data' / 'ground_truth'
            dst_images.mkdir(parents=True, exist_ok=True)
            dst_gt.mkdir(parents=True, exist_ok=True)

            print(f'\n=== {level_name} ===')

            # ground_truth：只链接本 level 实际需要的标注（硬链接默认不占额外空间）
            n_gt_linked, n_gt_missing = 0, 0
            link_mode = 'n/a'
            for p in img_paths:
                gt_name = p.name.replace('.jpg', '.mat').replace('img', 'GT_img')
                g_src = src_gt / gt_name
                if not g_src.is_file():
                    n_gt_missing += 1
                    continue
                link_mode = link_or_copy(g_src, dst_gt / gt_name,
                                         hardlink=not args.no_hardlink)
                n_gt_linked += 1
            msg = f'  ground_truth: {n_gt_linked} 个 .mat（{link_mode}）'
            if n_gt_missing:
                msg += f'  [警告] {n_gt_missing} 个标注缺失'
            print(msg)

            before, after, clipped = 0.0, 0.0, 0.0
            n_ok = 0
            cnt = [0]

            def work(p: Path):
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is None:
                    return None
                deg, clip = degrade(img, mode, alpha)
                dst = dst_images / p.name
                cv2.imwrite(str(dst), np.rint(deg).astype(np.uint8),
                            [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
                cnt[0] += 1
                if cnt[0] % 200 == 0 or cnt[0] == len(img_paths):
                    print(f'  {cnt[0]}/{len(img_paths)}', end='\r')
                return gray_mean(img), gray_mean(deg), clip

            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                for r in pool.map(work, img_paths):
                    if r is None:
                        continue
                    b, a, c = r
                    before += b
                    after += a
                    clipped += c
                    n_ok += 1

            print(f'  {cnt[0]}/{len(img_paths)} 完成' + ' ' * 12)

            if n_ok == 0:
                sys.exit(f'[错误] {level_name} 没有成功处理任何图像')

            before /= n_ok
            after /= n_ok
            clipped /= n_ok

            print(f'  平均亮度: {before:.2f} -> {after:.2f}  (比值 {after / before:.4f}，目标 alpha={alpha:g})')
            if clipped > 1e-9:
                print(f'  裁剪像素占比: {clipped * 100:.4f}%')

            stats_rows.append({
                'level': level_name, 'mode': mode, 'alpha': alpha, 'n_images': n_ok,
                'mean_brightness_before': round(before, 4),
                'mean_brightness_after': round(after, 4),
                'ratio': round(after / before, 6),
                'clipped_frac': round(clipped, 8),
                'data_path': str(level_root),
            })

    # ---- 汇总 ----
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / 'degradation_stats.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        w.writeheader()
        w.writerows(stats_rows)

    lines = [
        'Synthetic illumination degradation 汇总',
        '=' * 68,
        f'源数据目录 : {src_root}',
        f'图像数     : {len(img_paths)}',
        f'image_list : {args.image_list or "(全部)"}',
        f'JPEG 质量  : {args.jpeg_quality}',
        '',
        f'{"level":<18}{"before":>9}{"after":>9}{"ratio":>9}{"裁剪占比":>11}',
        '-' * 68,
    ]
    for r in stats_rows:
        lines.append(
            f'{r["level"]:<18}{r["mean_brightness_before"]:>9.2f}'
            f'{r["mean_brightness_after"]:>9.2f}{r["ratio"]:>9.4f}'
            f'{r["clipped_frac"] * 100:>10.4f}%'
        )
    lines += ['', '各 level 的 data_path（直接传给 main.py data_path=<dir>）:']
    for r in stats_rows:
        lines.append(f'  {r["mode"]:<8} alpha={r["alpha"]:<5g} -> {r["data_path"]}')

    summary = '\n'.join(lines)
    (out_root / 'summary.txt').write_text(summary, encoding='utf-8')
    print('\n' + summary)
    print(f'\n已写入 -> {out_root / "degradation_stats.csv"}')
    print(f'已写入 -> {out_root / "summary.txt"}')


if __name__ == '__main__':
    main()
