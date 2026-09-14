# -*- coding: utf-8 -*-
"""
对比 viz 与 vizgaijin 两个目录下同名图片的检测效果。
红色=预测结果, 绿色=真实标注。
对每张图: 提取红/绿掩码 -> 连通域 -> 中心点 -> 贪心最近邻匹配(距离阈值内)
计算 recall/precision/F1, 输出 vizgaijin 相比 viz 更好/更差的图片列表。
"""
import os
from collections import deque
from PIL import Image
import numpy as np

BASE = r"c:\Users\ZM\Desktop\uav-dot-main\results"
VIZ = os.path.join(BASE, "viz")
VIZG = os.path.join(BASE, "vizgaijin")
MATCH_DIST = 30.0  # 中心点匹配距离阈值(像素)


def get_masks(path):
    img = np.asarray(Image.open(path).convert("RGB")).astype(int)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    red = (R > 140) & (R - G > 50) & (R - B > 50)
    green = (G > 140) & (G - R > 50) & (G - B > 50)
    return red, green


def connected_centers(mask):
    """对掩码做 8 邻域连通域, 返回每个连通域的中心点 (y, x) 列表和大小。"""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    pts = set(zip(ys.tolist(), xs.tolist()))
    centers = []
    while pts:
        seed = pts.pop()
        comp = [seed]
        q = deque([seed])
        while q:
            y, x = q.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    nb = (y + dy, x + dx)
                    if nb in pts:
                        pts.remove(nb)
                        comp.append(nb)
                        q.append(nb)
        cy = sum(p[0] for p in comp) / len(comp)
        cx = sum(p[1] for p in comp) / len(comp)
        centers.append((cy, cx, len(comp)))
    return centers


def greedy_match(red_centers, green_centers, dist_thr):
    """贪心最近邻匹配。返回 (matched_green_idx, matched_red_idx, dists)"""
    pairs = []
    for gi, g in enumerate(green_centers):
        for ri, r in enumerate(red_centers):
            d = ((g[0] - r[0]) ** 2 + (g[1] - r[1]) ** 2) ** 0.5
            if d <= dist_thr:
                pairs.append((d, gi, ri))
    pairs.sort()
    used_g, used_r = set(), set()
    matched = []  # (gi, ri, d)
    for d, gi, ri in pairs:
        if gi in used_g or ri in used_r:
            continue
        used_g.add(gi)
        used_r.add(ri)
        matched.append((gi, ri, d))
    return matched


def evaluate(path):
    red, green = get_masks(path)
    rc = connected_centers(red)
    gc = connected_centers(green)
    matched = greedy_match(rc, gc, MATCH_DIST)
    tp = len(matched)
    n_g = len(gc)  # 真实目标数
    n_r = len(rc)  # 预测目标数
    recall = tp / n_g if n_g else 1.0
    precision = tp / n_r if n_r else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0
    # 像素级覆盖率: 真实(绿)像素中被预测(红)覆盖的比例
    red_px = red.sum()
    green_px = green.sum()
    overlap_px = int((red & green).sum())
    cov = overlap_px / green_px if green_px else 1.0
    return dict(recall=recall, precision=precision, f1=f1, n_g=n_g, n_r=n_r,
                tp=tp, cov=cov, overlap=overlap_px, red_px=int(red_px), green_px=int(green_px))


def main():
    names = sorted(os.listdir(VIZ))
    rows = []
    for name in names:
        p1 = os.path.join(VIZ, name)
        p2 = os.path.join(VIZG, name)
        if not os.path.exists(p2):
            continue
        e1 = evaluate(p1)
        e2 = evaluate(p2)
        rows.append((name, e1, e2))

    better, worse, same = [], [], []
    for name, e1, e2 in rows:
        d_f1 = e2["f1"] - e1["f1"]
        d_rec = e2["recall"] - e1["recall"]
        d_pre = e2["precision"] - e1["precision"]
        item = (name, e1, e2, d_f1, d_rec, d_pre)
        if d_f1 > 1e-9:
            better.append(item)
        elif d_f1 < -1e-9:
            worse.append(item)
        else:
            same.append(item)

    better.sort(key=lambda x: -x[3])
    worse.sort(key=lambda x: x[3])

    print(f"总图片数: {len(rows)}  阈值: {MATCH_DIST}px")
    print(f"vizgaijin 更好: {len(better)}  更差: {len(worse)}  相同: {len(same)}\n")

    print("=" * 100)
    print("【vizgaijin 更好的图片】(按 F1 改善从大到小)")
    print("=" * 100)
    print(f"{'图片':<28}{'viz R/P/F1':<24}{'gaijin R/P/F1':<24}{'dF1':<8}{'dRecall':<9}{'dPrec':<8}{'viz G/R':<10}{'gai G/R'}")
    for name, e1, e2, d_f1, d_rec, d_pre in better:
        print(f"{name:<28}{e1['recall']:.2f}/{e1['precision']:.2f}/{e1['f1']:.2f}   "
              f"{e2['recall']:.2f}/{e2['precision']:.2f}/{e2['f1']:.2f}   "
              f"{d_f1:+.3f}   {d_rec:+.3f}   {d_pre:+.3f}   "
              f"{e1['n_g']}/{e1['n_r']:<8}{e2['n_g']}/{e2['n_r']}")

    print()
    print("=" * 100)
    print("【vizgaijin 更差的图片】(按 F1 恶化从大到小)")
    print("=" * 100)
    for name, e1, e2, d_f1, d_rec, d_pre in worse:
        print(f"{name:<28}{e1['recall']:.2f}/{e1['precision']:.2f}/{e1['f1']:.2f}   "
              f"{e2['recall']:.2f}/{e2['precision']:.2f}/{e2['f1']:.2f}   "
              f"{d_f1:+.3f}   {d_rec:+.3f}   {d_pre:+.3f}   "
              f"{e1['n_g']}/{e1['n_r']:<8}{e2['n_g']}/{e2['n_r']}")


if __name__ == "__main__":
    main()
