#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按【绝对亮度阈值】把 DroneCrowd 测试集拆成 Low-light / Normal 两个文件夹。

划分流程
--------
    DroneCrowd Test
          │
    计算每个序列的平均亮度（归一化到 [0,1]）
          │
      绝对阈值 0.2
       /        \\
    < 0.2      >= 0.2
      │           │
   Low-light    Normal

拆完直接用原测试命令评测，只需把 data_path 指向对应文件夹。

用法
----
    python data/split_illumination.py --data-path D:/Dronecrowd/test_data \\
        --out-dir D:/Dronecrowd/splits

    # 需要其他阈值时（归一化亮度，0~1）
    python data/split_illumination.py --data-path ... --out-dir ... --threshold 0.33

输出（images 硬链接，几乎不额外占磁盘）
--------------------------------------
    D:/Dronecrowd/splits/
    ├── lowlight/test_data/{images,ground_truth}   # 序列平均亮度 <  0.2
    ├── normal/test_data/{images,ground_truth}     # 序列平均亮度 >= 0.2
    └── summary.txt                                # 阈值 + 序列清单 + 逐序列亮度表

评测
----
    python main.py --config-name=dronecrowd \\
        data_path=D:/Dronecrowd/splits/lowlight \\
        test_only=True debug=True \\
        restore_from_ckpt=/path/to/ckpt.ckpt

    python main.py --config-name=dronecrowd \\
        data_path=D:/Dronecrowd/splits/normal \\
        test_only=True debug=True \\
        restore_from_ckpt=/path/to/ckpt.ckpt

说明
----
* 分组按【序列级】平均亮度。DroneCrowd 的昼/夜是按序列连续分布的，逐帧划分会把
  同一场景的相邻帧拆到两组，引入场景差异的混淆因素。
* 亮度定义为灰度均值 / 255，与训练时的归一化尺度一致。
* .jpg 用硬链接（不占空间）；.mat 若带只读属性则复制（每个几百字节，总计几 MB），
  这样以后重新划分时不会因为只读属性删不掉。
"""

import argparse
import os
import shutil
import stat
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

DEFAULT_THRESHOLD = 0.2
FILE_ATTRIBUTE_READONLY = 0x1


# --------------------------------------------------------------------------- #
# 文件工具
# --------------------------------------------------------------------------- #
def norm_brightness(path: Path) -> float:
    """归一化亮度 = 灰度均值 / 255，取值 [0,1]。用 1/8 分辨率读取加速。"""
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    return float(img.mean()) / 255.0 if img is not None else 0.0


def is_readonly(path: Path) -> bool:
    """是否带 Windows 只读属性（非 Windows 返回 False）。"""
    try:
        return bool(path.stat().st_file_attributes & FILE_ATTRIBUTE_READONLY)
    except (AttributeError, OSError):
        return False


def clear_readonly(path: Path) -> bool:
    """清除只读属性，返回是否清除了。"""
    if not is_readonly(path):
        return False
    try:
        os.chmod(path, stat.S_IWRITE)
        return True
    except OSError:
        return False


def place(src: Path, dst: Path, hardlink: bool) -> str:
    """放置文件：优先硬链接，只读文件则复制（并去掉副本的只读属性）。"""
    if dst.exists():
        return 'exists'
    if hardlink and not is_readonly(src):
        try:
            os.link(src, dst)
            return 'hardlink'
        except OSError:
            pass
    shutil.copy2(src, dst)
    clear_readonly(dst)   # 副本去掉只读，方便下次重新划分时删除
    return 'copy'


def rmtree_force(path: Path) -> int:
    """删除目录树；遇到只读文件先清属性再删。返回处理过的只读文件数。

    注意：硬链接与源文件共享属性，因此清只读会同时影响源 .mat 的只读标记
    （只影响属性，不影响内容）。
    """
    fixed = [0]

    def _handle(func, p, exc):
        if clear_readonly(Path(p)):
            fixed[0] += 1
        try:
            func(p)
        except OSError:
            os.chmod(p, stat.S_IWRITE)
            func(p)

    try:
        shutil.rmtree(path, onexc=_handle)      # Python >= 3.12
    except TypeError:
        shutil.rmtree(path, onerror=_handle)    # Python < 3.12
    return fixed[0]


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description='按绝对亮度阈值把测试集拆成 lowlight / normal 两个文件夹')
    ap.add_argument('--data-path', required=True,
                    help='输入目录，需含 images/ 与 ground_truth/（如 D:/Dronecrowd/test_data）')
    ap.add_argument('--out-dir', required=True,
                    help='输出目录，生成 <out-dir>/lowlight 与 <out-dir>/normal')
    ap.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                    help=f'归一化亮度阈值（0~1），默认 {DEFAULT_THRESHOLD}')
    ap.add_argument('--copy', action='store_true', help='全部复制而非硬链接')
    ap.add_argument('--keep-old', action='store_true',
                    help='不清理输出目录中已有的同名分组目录（默认会清理）')
    ap.add_argument('--workers', type=int, default=8, help='读取图像的并发线程数')
    args = ap.parse_args()

    src_root, out_root = Path(args.data_path), Path(args.out_dir)
    src_images, src_gt = src_root / 'images', src_root / 'ground_truth'
    thr = args.threshold

    if not src_images.is_dir():
        sys.exit(f'[错误] 找不到图像目录: {src_images}')
    # 数据集代码靠字符串替换定位标注（images->ground_truth、img->GT_img），
    # 输出路径含 "img" 会让替换错位
    if 'img' in str(out_root).lower():
        sys.exit(f'[错误] 输出路径不能包含 "img": {out_root}')
    if src_root.resolve() == out_root.resolve():
        sys.exit('[错误] 输出目录不能与输入目录相同')

    img_paths = sorted(src_images.glob('*.jpg'))
    if not img_paths:
        sys.exit(f'[错误] {src_images} 下没有 .jpg 文件')
    print(f'共 {len(img_paths)} 张图像，计算亮度 ...')

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        bright = list(pool.map(norm_brightness, img_paths))

    def seq_of(p: Path) -> str:
        """序列号：img 后除末 3 位帧号以外的部分。"""
        d = p.stem[3:] if p.stem.startswith('img') else p.stem
        return d[:-3] if len(d) > 3 else d

    per_seq = {}
    for p, b in zip(img_paths, bright):
        per_seq.setdefault(seq_of(p), []).append(b)
    seq_avg = {s: float(np.mean(v)) for s, v in per_seq.items()}

    low_seqs = {s for s, v in seq_avg.items() if v < thr}
    low_set = {p for p in img_paths if seq_of(p) in low_seqs}
    groups = {
        'lowlight': [p for p in img_paths if p in low_set],
        'normal':   [p for p in img_paths if p not in low_set],
    }

    # ---- 写出文件 ----
    out_root.mkdir(parents=True, exist_ok=True)
    ro_fixed_total = 0
    for name in ('lowlight', 'normal'):
        group_dir = out_root / name
        if group_dir.exists() and not args.keep_old:
            n_fixed = rmtree_force(group_dir)
            ro_fixed_total += n_fixed
            note = f'（其中 {n_fixed} 个只读文件已清属性后删除）' if n_fixed else ''
            print(f'[清理] 移除旧目录 {group_dir}{note}')

        items = groups[name]
        if not items:
            print(f'[警告] {name} 组为空，跳过')
            continue

        dst_images = group_dir / 'test_data' / 'images'
        dst_gt = group_dir / 'test_data' / 'ground_truth'
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_gt.mkdir(parents=True, exist_ok=True)

        n_link = n_copy = 0
        missing = 0
        for p in items:
            if place(p, dst_images / p.name, not args.copy) == 'copy':
                n_copy += 1
            else:
                n_link += 1
            gt = src_gt / p.name.replace('.jpg', '.mat').replace('img', 'GT_img')
            if gt.is_file():
                if place(gt, dst_gt / gt.name, not args.copy) == 'copy':
                    n_copy += 1
                else:
                    n_link += 1
            else:
                missing += 1
        print(f'[{name}] 硬链接 {n_link} 个，复制 {n_copy} 个'
              + (f'，[警告] {missing} 个 .mat 缺失' if missing else ''))

    # ---- summary.txt ----
    seq_sorted = sorted(seq_avg.items(), key=lambda kv: kv[1])
    lines = [
        '光照划分结果（绝对亮度阈值）',
        '=' * 62,
        f'输入目录   : {src_root}',
        f'图像总数   : {len(img_paths)}',
        f'分组粒度   : 序列级平均亮度（灰度均值 / 255）',
        f'绝对阈值   : {thr:g}',
        '',
    ]
    for name in ('lowlight', 'normal'):
        items = groups[name]
        seqs = sorted({seq_of(p) for p in items})
        avg = float(np.mean([b for p, b in zip(img_paths, bright) if p in set(items)])) \
            if items else 0.0
        lines += [
            f'{"Low-light" if name == "lowlight" else "Normal":<10}: {len(items):>5} 帧 / '
            f'{len(seqs):>2} 序列   平均亮度 {avg:.4f}',
            f'  序列: {" ".join(seqs) if seqs else "(空)"}',
            f'  目录: {out_root / name}',
        ]

    lines += ['', '逐序列亮度（升序，* = Low-light）', '-' * 62,
              f'{"序列":<8}{"帧数":>7}{"平均亮度":>12}   分组']
    for seq, v in seq_sorted:
        mark = '*' if seq in low_seqs else ' '
        grp = 'Low-light' if seq in low_seqs else 'Normal'
        lines.append(f'{mark}{seq:<7}{len(per_seq[seq]):>7}{v:>12.4f}   {grp}')

    summary = '\n'.join(lines)
    (out_root / 'summary.txt').write_text(summary, encoding='utf-8')
    print()
    print(summary)
    print(f'\n评测时把 data_path 指向 {out_root}/lowlight 或 {out_root}/normal 即可。')


if __name__ == '__main__':
    main()
