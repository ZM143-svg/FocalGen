# -*- coding: utf-8 -*-
"""
对比修正预测(pt_pred)与原始预测(pt_pred51yuanshi)相对 ground_truth 的效果。
对每帧: 预测点与GT点做贪心最近邻匹配(距离阈值内算TP)。
指标: recall / precision / F1 / 匹配点平均距离。
输出: 修正后变好的帧、变差的帧、按序列汇总。
"""
import os

BASE = r"c:\Users\ZM\Desktop\uav-dot-main\results"
GT_DIR = os.path.join(BASE, "ground_truth")
ORI_DIR = os.path.join(BASE, "pt_pred51yuanshi", "merged_yolo_pixel")  # 原始预测
NEW_DIR = os.path.join(BASE, "pt_pred", "merged_yolo_pixel")           # 修正预测

THRESH = 30.0  # 匹配距离阈值(像素), 可调


def read_points(path):
    pts = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            pts.append((x, y))
    return pts


def greedy_match(pred, gt, thr):
    """贪心最近邻匹配, 每个GT点匹配最近的未用预测点。
    返回 (tp, fp, fn, matched_dists)"""
    used = [False] * len(pred)
    tp = 0
    dists = []
    for gx, gy in gt:
        best_i, best_d = -1, float("inf")
        for i, (px, py) in enumerate(pred):
            if used[i]:
                continue
            d = ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5
            if d < best_d:
                best_d, best_i = d, i
        if best_i >= 0 and best_d <= thr:
            used[best_i] = True
            tp += 1
            dists.append(best_d)
    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn, dists


def eval_frame(pred_pts, gt_pts, thr):
    tp, fp, fn, dists = greedy_match(pred_pts, gt_pts, thr)
    recall = tp / len(gt_pts) if gt_pts else 1.0
    precision = tp / len(pred_pts) if pred_pts else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0
    mean_dist = sum(dists) / len(dists) if dists else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, n_gt=len(gt_pts), n_pred=len(pred_pts),
                recall=recall, precision=precision, f1=f1, mean_dist=mean_dist)


def main():
    frames = []
    for name in sorted(os.listdir(GT_DIR)):
        gt_path = os.path.join(GT_DIR, name)
        base = os.path.splitext(name)[0]
        ori_path = os.path.join(ORI_DIR, base + "_loc.txt")
        new_path = os.path.join(NEW_DIR, base + "_loc.txt")
        if not (os.path.exists(ori_path) and os.path.exists(new_path)):
            print(f"[跳过] {name} 缺少对应预测文件")
            continue
        gt_pts = read_points(gt_path)
        ori_pts = read_points(ori_path)
        new_pts = read_points(new_path)
        e_ori = eval_frame(ori_pts, gt_pts, THRESH)
        e_new = eval_frame(new_pts, gt_pts, THRESH)
        frames.append((base, e_ori, e_new))

    better, worse, same = [], [], []
    for base, eo, en in frames:
        df = en["f1"] - eo["f1"]
        dd = en["mean_dist"] - eo["mean_dist"] if (eo["mean_dist"] == eo["mean_dist"] and en["mean_dist"] == en["mean_dist"]) else float("nan")
        item = (base, eo, en, df, dd)
        if df > 1e-9:
            better.append(item)
        elif df < -1e-9:
            worse.append(item)
        else:
            same.append(item)
    better.sort(key=lambda x: -x[3])
    worse.sort(key=lambda x: x[3])

    print(f"总帧数: {len(frames)}   匹配阈值: {THRESH}px")
    print(f"修正后 F1 变好: {len(better)} 帧   变差: {len(worse)} 帧   不变: {len(same)} 帧\n")

    # ---- 总体汇总 ----
    avg_f1_o = sum(eo["f1"] for _, eo, _ in frames) / len(frames)
    avg_f1_n = sum(en["f1"] for _, _, en in frames) / len(frames)
    ds = [en["mean_dist"] for _, _, en in frames if en["mean_dist"] == en["mean_dist"]]
    do = [eo["mean_dist"] for _, eo, _ in frames if eo["mean_dist"] == eo["mean_dist"]]
    avg_d_o = sum(do) / len(do) if do else float("nan")
    avg_d_n = sum(ds) / len(ds) if ds else float("nan")
    tot_tp_o = sum(eo["tp"] for _, eo, _ in frames)
    tot_tp_n = sum(en["tp"] for _, _, en in frames)
    tot_g = sum(eo["n_gt"] for _, eo, _ in frames)
    tot_p_o = sum(eo["n_pred"] for _, eo, _ in frames)
    tot_p_n = sum(en["n_pred"] for _, _, en in frames)
    print(f"总体(按帧平均) F1:  原始 {avg_f1_o:.4f}  ->  修正 {avg_f1_n:.4f}")
    print(f"总体(按帧平均) 匹配距离:  原始 {avg_d_o:.2f}px  ->  修正 {avg_d_n:.2f}px")
    print(f"总体(累计) TP:  原始 {tot_tp_o}/{tot_g}  ->  修正 {tot_tp_n}/{tot_g}")
    print(f"总体(累计) 预测点数:  原始 {tot_p_o}  ->  修正 {tot_p_n}\n")

    # ---- 按序列汇总 ----
    print("=" * 96)
    print("按序列汇总 (F1: 帧平均 | 距离: 匹配点平均, 像素)")
    print("=" * 96)
    from collections import defaultdict
    seq_o = defaultdict(list)
    seq_n = defaultdict(list)
    for base, eo, en in frames:
        s = base.split("_frame")[0]
        seq_o[s].append(eo)
        seq_n[s].append(en)
    print(f"{'序列':<14}{'帧数':<6}{'原始F1':<10}{'修正F1':<10}{'dF1':<10}{'原始距离':<12}{'修正距离':<12}{'dDist'}")
    for s in sorted(seq_o):
        eos, ens = seq_o[s], seq_n[s]
        f1o = sum(e["f1"] for e in eos) / len(eos)
        f1n = sum(e["f1"] for e in ens) / len(ens)
        do = [e["mean_dist"] for e in eos if e["mean_dist"] == e["mean_dist"]]
        dn = [e["mean_dist"] for e in ens if e["mean_dist"] == e["mean_dist"]]
        mo = sum(do) / len(do) if do else float("nan")
        mn = sum(dn) / len(dn) if dn else float("nan")
        dd = mn - mo if (mo == mo and mn == mn) else float("nan")
        print(f"{s:<14}{len(eos):<6}{f1o:<10.4f}{f1n:<10.4f}{f1n-f1o:<+10.4f}{mo:<12.2f}{mn:<12.2f}{dd:+.2f}")

    print()
    print("=" * 96)
    print(f"【修正后变好的帧】(共{len(better)}帧, 按 dF1 降序)")
    print("=" * 96)
    print(f"{'帧':<28}{'原始 R/P/F1':<22}{'修正 R/P/F1':<22}{'dF1':<8}{'TP原->新':<12}{'距离原->新'}")
    for base, eo, en, df, dd in better:
        dstr = f"{eo['mean_dist']:.1f}->{en['mean_dist']:.1f}" if (eo['mean_dist'] == eo['mean_dist'] and en['mean_dist'] == en['mean_dist']) else "n/a"
        print(f"{base:<28}{eo['recall']:.2f}/{eo['precision']:.2f}/{eo['f1']:.2f}   "
              f"{en['recall']:.2f}/{en['precision']:.2f}/{en['f1']:.2f}   "
              f"{df:+.3f}   {eo['tp']}->{en['tp']:<8}{dstr}")

    print()
    print("=" * 96)
    print(f"【修正后变差的帧】(共{len(worse)}帧, 按 dF1 升序)")
    print("=" * 96)
    for base, eo, en, df, dd in worse:
        dstr = f"{eo['mean_dist']:.1f}->{en['mean_dist']:.1f}" if (eo['mean_dist'] == eo['mean_dist'] and en['mean_dist'] == en['mean_dist']) else "n/a"
        print(f"{base:<28}{eo['recall']:.2f}/{eo['precision']:.2f}/{eo['f1']:.2f}   "
              f"{en['recall']:.2f}/{en['precision']:.2f}/{en['f1']:.2f}   "
              f"{df:+.3f}   {eo['tp']}->{en['tp']:<8}{dstr}")


if __name__ == "__main__":
    main()
