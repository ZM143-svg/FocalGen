# import math
#
# import matplotlib.pyplot as plt
# import os
#
# import torch
# from sympy import false
# from torch import nn
#
#
# # class StableInstanceNorm(nn.Module):
# #     """稳定的实例归一化实现"""
# #
# #     def __init__(self, channels, epsilon=1e-3):
# #         super(StableInstanceNorm, self).__init__()
# #         self.epsilon = epsilon
# #         self.channels = channels
# #         # 可学习参数 γ 和 β
# #         self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))  # 缩放参数
# #         self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))  # 偏移参数
# #
# #     def forward(self, x):
# #         B, C, H, W = x.shape
# #
# #         # 检查输入是否异常
# #         if torch.isnan(x).any() or torch.isinf(x).any():
# #             print("StableInstanceNorm: Input contains NaN/Inf, returning identity")
# #             return x
# #
# #         # 手动计算均值和方差
# #         mean = x.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
# #         var = x.var(dim=(2, 3), keepdim=True, unbiased=False)  # [B, C, 1, 1]
# #
# #         # 稳定性处理
# #         var = torch.clamp(var, min=self.epsilon ** 2, max=1e6)
# #         std = torch.sqrt(var+ 1e-8)
# #
# #         # 归一化并应用可学习参数
# #         # x' = γ * (x - μ) / σ + β
# #         normalized = self.gamma * (x - mean) / std + self.beta
# #         # 检查输出是否异常
# #         if torch.isnan(normalized).any() or torch.isinf(normalized).any():
# #             print("StableInstanceNorm: Output contains NaN/Inf, returning identity")
# #             return x
# #
# #         return normalized
#
# class CSNorm(nn.Module):
#     """Channel Selective Normalization module with visualization"""
#
#     def __init__(self, channels,  name="unknown", epsilon=1e-5, visualize_interval=200):
#         super(CSNorm, self).__init__()
#         self.channels = channels
#         self.name = name
#         self.epsilon = epsilon
#         self.visualize_interval = visualize_interval
#         self.step_counter = 0
#
#         # 使用InstanceNorm但添加详细监控
#         self.instance_norm = nn.InstanceNorm2d(channels, affine=True)
#
#         # 可微门控模块
#         self.gating_module = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, max(1, channels // 4), 1),
#             nn.ReLU(),
#             nn.Dropout(0.1),  # 添加dropout防止过拟合
#             nn.Conv2d(max(1, channels // 4), channels, 1),
#             nn.Sigmoid()
#         )
#         # 更好的初始化策略
#         self._initialize_weights()
#
#     # def _initialize_weights(self):
#     #     """改进的权重初始化"""
#     #     for m in self.gating_module.modules():
#     #         if isinstance(m, nn.Conv2d):
#     #             # 使用Xavier初始化，适合Sigmoid
#     #             nn.init.xavier_normal_(m.weight)
#     #             if m.bias is not None:
#     #                 nn.init.constant_(m.bias, 0)
#     #
#     #     # 特别初始化最后一个卷积层，使其初始输出接近0.3
#     #     last_conv = self.gating_module[-2]
#     #     nn.init.normal_(last_conv.weight, mean=0.0, std=0.1)
#     #     nn.init.constant_(last_conv.bias, -1.0)  # 经过Sigmoid后输出约0.27
#     #
#     #     # # 创建可视化目录
#     #     # self.viz_dir = f"csnorm_visualization/{name}"
#     #     # os.makedirs(self.viz_dir, exist_ok=True)
#
#     def _initialize_weights(self):
#         # 更精确的初始化，促进早期学习
#         for m in self.gating_module.modules():
#             if isinstance(m, nn.Conv2d):
#                 # 使用Kaiming初始化，适合ReLU
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#         # 门控模块偏向于选择更多通道进行归一化（促进泛化）
#         last_conv = self.gating_module[-2]
#         nn.init.normal_(last_conv.weight, mean=0.0, std=0.1)  # 更小的方差
#         # 原始sigmoid(-1.0) = 0.269（归一化概率）
#         # p_norm = 0.269  # 原始归一化概率
#         # p_not_norm = 1 - p_norm  # 0.731
#         #
#         # # 计算对应的logits
#         # logit_not_norm = math.log(p_not_norm)  # log(0.731) ≈ -0.314
#         # logit_norm = math.log(p_norm)  # log(0.269) ≈ -1.314
#         #
#         # with torch.no_grad():
#         #     # 不归一化logit（偶数索引）
#         #     last_conv.bias[::2] = logit_not_norm
#         #     # 归一化logit（奇数索引）
#         #     last_conv.bias[1::2] = logit_norm
#
#         nn.init.constant_(last_conv.bias, -1.0)  # 初始偏向于激活（选择归一化）,百分之50
#
#     def forward(self, x):
#         normalized_x = self.instance_norm(x)
#         gate = self.gating_module(x)
#         # 使用Gumbel-Softmax进行二值化
#         # 形状调整为 [B, C, 1, 1] -> [B*C] 的二分类
#         # B, C, H, W = x.shape
#         # # 重塑为 [B, C, 2]
#         # logits = gate.view(B, 2, C, 1, 1).permute(0, 2, 1, 3, 4).squeeze(-1).squeeze(-1)  # [B, C, 2]
#         #
#         # # Gumbel-Softmax
#         # if self.training:
#         #     mask = torch.nn.functional.gumbel_softmax(
#         #         logits,
#         #         tau=1,
#         #         hard=True,
#         #         dim=2
#         #     )[:, :, 1]  # 形状 [B, C]
#         # else:
#         #     mask = (logits.argmax(dim=2) == 1).float()  # 形状 [B, C]
#         #
#         # # 重塑为特征图形状
#         # mask = mask.view(B, C, 1, 1)
#         #
#         # # 二值化混合
#         # output = mask * normalized_x + (1 - mask) * x
#         # # 记录统计信息（可选）
#         # # if self.training and torch.rand(1) < 0.01:
#         # #     norm_ratio = mask.mean().item()
#         # #     print(f"BinaryCSNorm: 归一化通道比例={norm_ratio:.2%}")
#         #
#         # return output
#         # # 改进2：添加温度控制，使门控更精确
#         # temperature = torch.clamp(self.temperature, 1.0, 10.0)
#         # gate = torch.sigmoid(temperature * (gate - 0.5))  # 锐化门控
#         output = (1 - gate) * x + gate * normalized_x
#         #
#         #
#         return output
#         #
import matplotlib.pyplot as plt
import os

import torch
from sympy import false
from torch import nn


# class StableInstanceNorm(nn.Module):
#     """稳定的实例归一化实现"""
#
#     def __init__(self, channels, epsilon=1e-3):
#         super(StableInstanceNorm, self).__init__()
#         self.epsilon = epsilon
#         self.channels = channels
#         # 可学习参数 γ 和 β
#         self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))  # 缩放参数
#         self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))  # 偏移参数
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#
#         # 检查输入是否异常
#         if torch.isnan(x).any() or torch.isinf(x).any():
#             print("StableInstanceNorm: Input contains NaN/Inf, returning identity")
#             return x
#
#         # 手动计算均值和方差
#         mean = x.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
#         var = x.var(dim=(2, 3), keepdim=True, unbiased=False)  # [B, C, 1, 1]
#
#         # 稳定性处理
#         var = torch.clamp(var, min=self.epsilon ** 2, max=1e6)
#         std = torch.sqrt(var+ 1e-8)
#
#         # 归一化并应用可学习参数
#         # x' = γ * (x - μ) / σ + β
#         normalized = self.gamma * (x - mean) / std + self.beta
#         # 检查输出是否异常
#         if torch.isnan(normalized).any() or torch.isinf(normalized).any():
#             print("StableInstanceNorm: Output contains NaN/Inf, returning identity")
#             return x
#
#         return normalized

class CSNorm(nn.Module):
    """Channel Selective Normalization module with visualization"""

    def __init__(self, channels,  name="unknown", epsilon=1e-5):
        super(CSNorm, self).__init__()
        self.channels = channels
        self.name = name
        self.epsilon = epsilon
        # self.visualize_interval = visualize_interval
        # self.step_counter = 0

        # 使用InstanceNorm但添加详细监控 visualize_interval=200
        self.instance_norm = nn.InstanceNorm2d(channels, affine=True)

        # 可微门控模块
        self.gating_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(1, channels // 4), 1),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加dropout防止过拟合
            nn.Conv2d(max(1, channels // 4), channels, 1),
            nn.Sigmoid()
        )
        # 推理时的阈值（默认0.5，可在验证集上调优）
        # self.inference_threshold = 0.5
        # # 更好的初始化策略
        # self._initialize_weights()

    # def _initialize_weights(self):
    #     """改进的权重初始化"""
    #     for m in self.gating_module.modules():
    #         if isinstance(m, nn.Conv2d):
    #             # 使用Xavier初始化，适合Sigmoid
    #             nn.init.xavier_normal_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #
    #     # 特别初始化最后一个卷积层，使其初始输出接近0.3
    #     last_conv = self.gating_module[-2]
    #     nn.init.normal_(last_conv.weight, mean=0.0, std=0.1)
    #     nn.init.constant_(last_conv.bias, -1.0)  # 经过Sigmoid后输出约0.27
    #
    #     # # 创建可视化目录
    #     # self.viz_dir = f"csnorm_visualization/{name}"
    #     # os.makedirs(self.viz_dir, exist_ok=True)

    def _initialize_weights(self):
        # 更精确的初始化，促进早期学习
        for m in self.gating_module.modules():
            if isinstance(m, nn.Conv2d):
                # 使用Kaiming初始化，适合ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 门控模块偏向于选择更多通道进行归一化（促进泛化）
        last_conv = self.gating_module[-2]
        nn.init.normal_(last_conv.weight, mean=0.0, std=0.02)  # 更小的方差
        nn.init.constant_(last_conv.bias, 0)  # 初始偏向于激活（选择归一化）

    def forward(self, x):
        normalized_x = self.instance_norm(x)
        gate = self.gating_module(x)

        # # 改进2：添加温度控制，使门控更精确
        # temperature = torch.clamp(self.temperature, 1.0, 10.0)
        # gate = torch.sigmoid(temperature * (gate - 0.5))  # 锐化门控
        output = (1 - gate) * x + gate * normalized_x


        return output

    # def set_threshold(self, threshold):
    #     """设置推理阈值（在验证集上调优后调用）"""
        self.inference_threshold = threshold
# class CSNorm(nn.Module):
#     """Channel Selective Normalization module (exactly as described in the paper)"""
#     def __init__(self, channels, name="unknown", epsilon=1e-5):
#         super(CSNorm, self).__init__()
#         self.channels = channels
#         self.name = name
#         self.epsilon = epsilon

#         # Instance Normalization (with affine params)
#         self.instance_norm = nn.InstanceNorm2d(channels, affine=True)

#         # Gating module: outputs alpha_x (not sigmoid)
#         self.gating_module = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, max(1, channels // 4), 1),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Conv2d(max(1, channels // 4), channels, 1)   # No activation, raw alpha
#         )

#         self._initialize_weights()

#     def _initialize_weights(self):
#         # Initialize as in paper: small weights, zero bias to keep alpha near 0 initially
#         for m in self.gating_module.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#         # Make the last layer produce near-zero alpha to start with (g≈0)
#         last_conv = self.gating_module[-1]
#         nn.init.normal_(last_conv.weight, mean=0.0, std=0.02)
#         nn.init.constant_(last_conv.bias, 0.0)

#     def forward(self, x):
#         # Normalize
#         x_norm = self.instance_norm(x)

#         # Compute alpha_x (shape: B, C, 1, 1)
#         alpha = self.gating_module(x)           # (B, C, 1, 1)
#         alpha = alpha.view(alpha.size(0), alpha.size(1))   # (B, C)

#         # Hard gating: g = alpha^2 / (alpha^2 + epsilon)
#         alpha_sq = alpha ** 2
#         g = alpha_sq / (alpha_sq + self.epsilon)   # (B, C) → values close to 0 or 1

#         # Reshape for broadcasting
#         g = g.view(g.size(0), g.size(1), 1, 1)     # (B, C, 1, 1)

#         # Output: (1-g)*x + g*x_norm
#         output = (1 - g) * x + g * x_norm
#         return output
