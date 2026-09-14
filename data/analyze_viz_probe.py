# -*- coding: utf-8 -*-
"""探查 viz 图片中的红/绿像素颜色特征，验证颜色假设。"""
import os
from PIL import Image
import numpy as np

VIZ = r"c:\Users\ZM\Desktop\uav-dot-main\results\viz"

def count_colors(path):
    img = np.asarray(Image.open(path).convert("RGB")).astype(int)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    red = (R > 150) & (R - G > 50) & (R - B > 50)
    green = (G > 150) & (G - R > 50) & (G - B > 50)
    return img.shape, red.sum(), green.sum()

for name in ["dataset1_frame_00001.jpg", "dataset11_frame_00020.jpg", "dataset12_frame_00001.jpg"]:
    p = os.path.join(VIZ, name)
    print(name, count_colors(p))

# 打印红/绿像素的颜色分布样本
img = np.asarray(Image.open(os.path.join(VIZ, "dataset1_frame_00001.jpg")).convert("RGB")).astype(int)
R, G, B = img[..., 0], img[..., 1], img[..., 2]
red_mask = (R > 150) & (R - G > 50) & (R - B > 50)
green_mask = (G > 150) & (G - R > 50) & (G - B > 50)
if red_mask.sum():
    print("red sample colors:", img[red_mask][::max(1, red_mask.sum() // 5)])
if green_mask.sum():
    print("green sample colors:", img[green_mask][::max(1, green_mask.sum() // 5)])
