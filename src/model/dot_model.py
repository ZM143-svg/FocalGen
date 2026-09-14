import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead
from src.model.dysample import DySample  # 假设 dysample.py 放在同一目录或已加入 sys.path
from src.model.CSNorm import CSNorm


class PDAttentionModule(nn.Module):
    """基于PD输出的注意力掩码生成模块，使用软阈值回归和排名损失监督"""

    def __init__(self, in_channels=3, feat_channels=32,
                 dense_thresh=0.3, sparse_thresh=0.005,
                 dense_range=(1.2, 1.6), sparse_range=(0.8, 1.2),
                 background_range=(0.0, 0.2),
                 adaptive_weight=False):
        super().__init__()

        # 存储参数
        self.adaptive_weight = adaptive_weight
        if adaptive_weight:
            # 可学习的权重参数
            self.dense_weight = nn.Parameter(torch.tensor(1.5))
            self.sparse_weight = nn.Parameter(torch.tensor(1.5))
            self.bg_weight = nn.Parameter(torch.tensor(0.2))
            self.rank_weight = nn.Parameter(torch.tensor(0.1))

        self.dense_thresh = dense_thresh
        self.sparse_thresh = sparse_thresh
        self.dense_range = dense_range
        self.sparse_range = sparse_range
        self.background_range = background_range

        # 轻量级特征提取，从PD输出生成注意力
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels // 2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(feat_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels // 2, feat_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )

        # 注意力生成
        self.attention_generator = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels // 2, 1),
            nn.BatchNorm2d(feat_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels // 2, 1, 1),
            nn.Softplus()  # 输出自由
        )

        # # 可学习的缩放因子，将[0,1]映射到合适的范围
        # self.scale_factor = nn.Parameter(torch.tensor(1.6))  # 初始化为最大范围

        # 排名损失的边际
        self.rank_margin = nn.Parameter(torch.tensor(0.2))
        self.temperature = 5.0  # 控制过渡带陡峭度

    def forward(self, pd_feature, gt_density=None):
        """
        pd_feature: PD模块的输出特征 (B, C, H, W)
        gt_density: 高斯热图监督 (B, 1, H, W)，训练时提供，测试时为None
        """
        # 保存原始尺寸
        original_size = pd_feature.shape[-2:]

        # 提取特征并生成基础注意力图（范围[0,1]）
        features = self.feature_extractor(pd_feature)
        base_attention = self.attention_generator(features)

        # 动态调整到原始尺寸
        base_attention = F.interpolate(base_attention, size=original_size,
                                       mode='bilinear', align_corners=True)

        # # 应用缩放因子，将注意力映射到[0, scale_factor]范围
        # # 使用softplus确保缩放因子为正,不用缩放
        # scale = F.softplus(self.scale_factor)
        # attention_map = base_attention * scale

        # 训练时计算监督损失
        supervised_loss = torch.tensor(0.0, device=pd_feature.device)
        if gt_density is not None and self.training:
            supervised_loss = self._compute_supervised_loss(
                base_attention, gt_density
            )

        return base_attention, supervised_loss

    def _compute_supervised_loss(self, attention_map, gt_density):
        """
        计算基于软阈值和排名损失的监督损失 - 稳健版本
        attention_map: 生成的注意力图 (B, 1, H, W)
        gt_density: 高斯热图监督 (B, 1, H, W)
        """
        batch_size = attention_map.shape[0]
        total_loss = torch.tensor(0.0, device=attention_map.device)

        if attention_map.shape[-2:] != gt_density.shape[-2:]:
            gt_density_resize = F.interpolate(gt_density, size=attention_map.shape[-2:],
                                              mode='bilinear', align_corners=True)
        else:
            gt_density_resize = gt_density

        for b in range(batch_size):
            density = gt_density_resize[b]  # (1, H, W)
            attn = attention_map[b]  # (1, H, W)

            # 修正：使用 density 而不是 gt_density
            dense_mask = torch.sigmoid(self.temperature * (density - self.dense_thresh))
            sparse_mask = torch.sigmoid(self.temperature * (density - self.sparse_thresh)) * \
                          (1.0 - torch.sigmoid(self.temperature * (density - self.dense_thresh)))
            background_mask = 1.0 - torch.sigmoid(self.temperature * (density - self.sparse_thresh))

            if dense_mask.sum() > 0:
                N_dense = dense_mask.sum()
                min_val = dense_mask.min()
                max_val = dense_mask.max()
                #
                # # 软目标：将密度值线性映射到[1.2, 1.6]
                # density_scaled = density * (self.dense_range[1] - self.dense_range[0]) / \
                #                  (density.max() + 1e-8) + self.dense_range[0]
                if max_val - min_val > 1e-8:
                    dense_target = 1.2 + 0.4 * (dense_mask - min_val) / (max_val - min_val)
                else:
                    dense_target = torch.ones_like(dense_mask) * 1.0

                dense_loss = (dense_mask * F.smooth_l1_loss(attn, dense_target, reduction='none', beta=0.1)).sum() / (
                            N_dense + 1e-8)
            else:
                dense_loss = torch.tensor(0.0, device=attn.device)

            # # 3. 固定目标值（更稳定）
            # dense_target = 1.6
            # # 密集区域损失：鼓励高注意力
            # if dense_mask.sum() > 0:
            #     dense_loss = F.mse_loss(attn[dense_mask],
            #                             torch.ones_like(attn[dense_mask]) * dense_target)
            # else:
            #     dense_loss = torch.tensor(0.0, device=attn.device)
            #
            # # # 稀疏区域损失：适度监督
            # # if dense_mask.sum() > 0:
            # #     sparse_loss = (sparse_mask * torch.abs(attn - dense_target)).sum() / (sparse_mask.sum() + 1e-8)
            # #     sparse_loss = sparse_loss
            # # else:
            # #     sparse_loss = torch.tensor(0.0, device=attention_map.device)

            if sparse_mask.sum() > 0:
                # 将密度线性映射到[0.8, 1.2]
                N_sparse = sparse_mask.sum()
                min_val = sparse_mask.min()
                max_val = sparse_mask.max()

                if max_val - min_val > 1e-8:
                    sparse_target = 0.8 + 0.4 * (sparse_mask - min_val) / (max_val - min_val)
                else:
                    sparse_target = torch.ones_like(sparse_mask) * 1.0

                sparse_loss = (sparse_mask * F.smooth_l1_loss(attn, sparse_target, reduction='none',
                                                              beta=0.1)).sum() / (N_sparse + 1e-8)
            else:
                sparse_loss = torch.tensor(0.0, device=attn.device)

            if background_mask.sum() > 0:
                bg_target = 0.1  # 固定值<=0.1
                background_loss = (background_mask * torch.nn.functional.relu(attn - bg_target)).sum() / (
                            background_mask.sum() + 1e-8)
            else:
                background_loss = torch.tensor(0.0, device=attn.device)

            # # 稀疏区域损失：适度监督
            # if sparse_mask.sum() > 0:
            #     sparse_loss = (sparse_mask * torch.abs(attn - sparse_target)).sum() / (sparse_mask.sum() + 1e-8)
            #     sparse_loss = sparse_loss
            # else:
            #     sparse_loss = torch.tensor(0.0, device=attention_map.device)

            # # 2. 计算各区域损失 - 使用稳健的逆频率权重
            # N = density.numel()  # 总像素数
            #
            # # 密集区域损失
            # if dense_mask.sum() > 0:
            #     N_dense = dense_mask.sum()
            #     # # 方法1：使用平方根逆频率权重（推荐）
            #     # w_dense = torch.sqrt(N / (N_dense + 1e-8))
            #     # 或者方法2：使用对数逆频率权重（更平滑）
            #     # w_dense = torch.log(N / (N_dense + 1e-8) + 1)
            #
            #     # 软目标：将密度值线性映射到[1.2, 1.6]
            #     density_scaled = density * (self.dense_range[1] - self.dense_range[0]) / \
            #                      (density.max() + 1e-8) + self.dense_range[0]
            #
            #     # 使用Huber损失（对异常值更鲁棒）或MSE
            #     dense_loss = (dense_mask * F.smooth_l1_loss(attn, density_scaled, reduction='none', beta=0.1)).sum() / \
            #                  (N_dense + 1e-8)
            #     dense_loss = dense_loss
            # else:
            #     dense_loss = torch.tensor(0.0, device=attention_map.device)
            #
            # # 稀疏区域损失
            # if sparse_mask.sum() > 0:
            #     N_sparse = sparse_mask.sum()
            #     # w_sparse = torch.sqrt(N / (N_sparse + 1e-8))
            #
            #     density_scaled = density * (self.sparse_range[1] - self.sparse_range[0]) / \
            #                      (density.max() + 1e-8) + self.sparse_range[0]
            #
            #     sparse_loss = (sparse_mask * F.smooth_l1_loss(attn, density_scaled, reduction='none', beta=0.1)).sum() / \
            #                   (N_sparse + 1e-8)
            #     sparse_loss = sparse_loss
            # else:
            #     sparse_loss = torch.tensor(0.0, device=attention_map.device)
            #
            # 背景区域损失
            # if background_mask.sum() > 0:
            #     N_background = background_mask.sum()
            #     # 背景区域权重：适当降低但不为负
            #     # w_background = torch.sqrt(N_background / (N + 1e-8))  # [0, 1]范围
            #
            #     density_scaled = density * (self.background_range[1] - self.background_range[0]) / \
            #                      (density.max() + 1e-8) + self.background_range[0]
            #
            #     background_loss = (background_mask * F.smooth_l1_loss(attn, density_scaled, reduction='none',
            #                                                           beta=0.1)).sum() / \
            #                       (N_background + 1e-8)
            #     background_loss = background_loss
            # else:
            #     background_loss = torch.tensor(0.0, device=attention_map.device)

            # 3. 排名损失 - 确保高密度区域注意力 > 低密度区域注意力
            density_flat = density.view(-1)
            attn_flat = attn.view(-1)

            high_indices = torch.where(density_flat > self.dense_thresh)[0]
            low_indices = torch.where(density_flat < self.sparse_thresh)[0]

            rank_loss = torch.tensor(0.0, device=attention_map.device)
            if len(high_indices) > 0 and len(low_indices) > 0:
                n_pairs = min(100, len(high_indices), len(low_indices))
                high_samples = high_indices[torch.randperm(len(high_indices))[:n_pairs]]
                low_samples = low_indices[torch.randperm(len(low_indices))[:n_pairs]]

                high_attn = attn_flat[high_samples]
                low_attn = attn_flat[low_samples]

                # 使用软边际排名损失
                margin = F.softplus(self.rank_margin)
                # 使用对数间隔排名损失，对大小差异更敏感
                rank_loss = F.relu(torch.log(low_attn + 1e-8) - torch.log(high_attn + 1e-8) + margin).mean()

            # # 4. 稀疏性正则化：鼓励注意力图简洁但不过度稀疏
            # sparsity_loss = torch.abs(attn.mean() - 0.3) * 0.01  # 鼓励平均注意力在0.3左右
            #
            # # 5. 一致性正则化：鼓励相似密度区域有相似注意力
            # consistency_loss = torch.tensor(0.0, device=attention_map.device)
            # if sparse_mask.sum() > 0:
            #     # 计算稀疏区域内注意力的标准差，鼓励一致性
            #     sparse_attn = attn * sparse_mask
            #     sparse_mean = sparse_attn.sum() / (sparse_mask.sum() + 1e-8)
            #     sparse_std = ((sparse_attn - sparse_mean) ** 2 * sparse_mask).sum() / (sparse_mask.sum() + 1e-8)
            #     consistency_loss = sparse_std * 0.01

            # 6. 总损失 - 带权重调整
            # 权重分配：密集区域最重要，稀疏区域次之，排名损失辅助
            # 自适应权重
            if self.adaptive_weight:
                # 使用softplus确保权重为正
                w_dense = F.softplus(self.dense_weight)
                w_sparse = F.softplus(self.sparse_weight)
                w_bg = F.softplus(self.bg_weight)
                w_rank = F.softplus(self.rank_weight)
            else:
                w_dense, w_sparse, w_bg, w_rank = 1.5, 1.5, 0.2, 0.1

            total_loss += (w_dense * dense_loss + w_sparse * sparse_loss +
                           w_bg * background_loss + w_rank * rank_loss)

        return total_loss / batch_size

    def apply_attention(self, x, attention_map, mode='adaptive'):
        """
        应用注意力到特征
        mode: multiply（直接乘）或 adaptive（自适应）
        """
        attention_map = torch.clamp(attention_map, min=0.0, max=3.0)  # 限制范围

        if mode == 'multiply':
            # 直接乘法：x = x * attention_map
            return x * attention_map
        elif mode == 'adaptive':
            # 自适应：根据注意力强度动态调整
            attention_strength = attention_map.mean(dim=[1, 2, 3], keepdim=True)
            alpha = 0.2 + 0.3 * attention_strength  # [0.2, 0.5]
            return x * (1 + attention_map * alpha)
        else:
            return x


class DynamicConv2d(nn.Module):
    """轻量动态卷积：分组 + 少专家（默认 num_experts=2, groups=4）"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 num_experts=2, groups=4, reduction=16, bias=False):
        super().__init__()
        assert in_channels % groups == 0 and out_channels % groups == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.groups = groups

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                      padding=padding, groups=groups, bias=bias)
            for _ in range(num_experts)
        ])

        reduced_dim = max(in_channels // reduction, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, reduced_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, num_experts * groups),
            nn.Softmax(dim=1)
        )
        self._init_weights()

    def _init_weights(self):
        for i in range(1, self.num_experts):
            self.convs[i].weight.data.copy_(self.convs[0].weight.data)

    def forward(self, x):
        B, C, H, W = x.shape
        gate_weights = self.gate(x).view(B, self.groups, self.num_experts)  # (B, groups, num_experts)
        out = torch.zeros(B, self.out_channels, H, W, device=x.device, dtype=x.dtype)
        for e in range(self.num_experts):
            conv_out = self.convs[e](x)
            weight = gate_weights[:, :, e].view(B, self.groups, 1, 1)
            weight = weight.repeat(1, self.out_channels // self.groups, 1, 1).view(B, self.out_channels, 1, 1)
            out += weight * conv_out
        return out


class DySampleDecoderBlock(nn.Module):
    """
    解码器块：使用 DySample 上采样 + 可选动态卷积
    - use_dynamic_conv: True 使用 DynamicConv2d，False 使用普通 Conv2d+BN+ReLU
    """

    def __init__(self,
                 in_channels,
                 skip_channels,
                 out_channels,
                 use_batchnorm=True,
                 attention_type=None,
                 use_dynamic_conv=False,  # 动态卷积开关
                 num_experts=4,  # 动态卷积专家数
                 dysample_style='lp',  # DySample 风格 'lp' 或 'pl'
                 dysample_groups=4,  # DySample 分组
                 dysample_dyscope=False):  # DySample 动态范围
        super().__init__()

        # 1. DySample 上采样器（scale=2 固定）
        self.upsample = DySample(
            in_channels=in_channels,
            scale=2,
            style=dysample_style,
            groups=dysample_groups,
            dyscope=dysample_dyscope
        )

        # 2. 第一个卷积层（可选动态卷积）
        conv1_in = in_channels + skip_channels
        if use_dynamic_conv:
            self.conv1 = nn.Sequential(
                DynamicConv2d(conv1_in, out_channels, kernel_size=3, padding=1,
                              num_experts=num_experts, bias=not use_batchnorm),
                nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv1 = md.Conv2dReLU(
                conv1_in, out_channels, kernel_size=3, padding=1,
                use_norm=use_batchnorm
            )

        # 3. 注意力（可选）
        self.attention1 = md.Attention(attention_type, in_channels=conv1_in) if attention_type else nn.Identity()

        # 4. 第二个卷积层（可选动态卷积）
        if use_dynamic_conv:
            self.conv2 = nn.Sequential(
                DynamicConv2d(out_channels, out_channels, kernel_size=3, padding=1,
                              num_experts=num_experts, bias=not use_batchnorm),
                nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv2 = md.Conv2dReLU(
                out_channels, out_channels, kernel_size=3, padding=1,
                use_norm=use_batchnorm
            )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels) if attention_type else nn.Identity()

    def forward(self, x, skip=None):
        # DySample 上采样
        x = self.upsample(x)
        # 拼接跳跃连接
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        # 卷积
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        _, _, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)

        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
            use_dysample=False,  # 是否使用 DySample + 可选动态卷积
            use_dynamic_conv=False,  # 动态卷积开关（仅当 use_dysample=True 时生效）
            num_experts=4,
            dysample_style='lp',
            dysample_groups=4,
            dysample_dyscope=False
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm,
                      attention_type=attention_type)
        # 根据 use_dysample 选择 DecoderBlock 类型
        if use_dysample:
            BlockClass = DySampleDecoderBlock
            extra_kwargs = {
                'use_dynamic_conv': use_dynamic_conv,
                'num_experts': num_experts,
                'dysample_style': dysample_style,
                'dysample_groups': dysample_groups,
                'dysample_dyscope': dysample_dyscope
            }
        else:
            BlockClass = DecoderBlock
            extra_kwargs = {}

        blocks = [
            BlockClass(in_ch, skip_ch, out_ch,
                       use_batchnorm=use_batchnorm,
                       attention_type=attention_type,
                       **extra_kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

        # blocks = [
        #     DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
        #     for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        # ]
        # self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        # remove first skip with same spatial resolution
        features = features[1:]
        # reverse channels to start from head of encoder
        features = features[::-1]

        head = features[0]
        skips = features[1:]

        x = self.center(head)

        out = []
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            out.append(x)

        return out[::-1]


class PixelDistillBlock(nn.Module):
    def __init__(self, in_channels, features_channels, scale_down=1):
        super(PixelDistillBlock, self).__init__()

        if scale_down > 1:
            self.down = nn.PixelUnshuffle(scale_down)
            in_channels = int(in_channels * scale_down ** 2)
        else:
            self.down = nn.Identity()

        self.conv = nn.Conv2d(in_channels, features_channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(features_channels)
        self.relu = nn.ReLU()

        self.attn = CoordAtt(inp=features_channels, oup=features_channels)

    def forward(self, x):
        x = self.down(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.attn(x)

        return x


class PixelDistill(nn.Module):
    def __init__(self, in_channels, out_channels, features_channels, down_scale=1):
        super(PixelDistill, self).__init__()
        self.down_reduction = int(2 ** 2)

        self.unshuffle = nn.PixelUnshuffle(2)

        self.blocks = nn.ModuleList([
            PixelDistillBlock(in_channels, features_channels // 2, scale_down=down_scale // 2) for _ in
            range(self.down_reduction)
        ])

        self.conv_1 = nn.Conv2d(int(self.down_reduction * (features_channels // 2)), features_channels, 3, 1, 1)
        self.bn_1 = nn.BatchNorm2d(features_channels)
        self.relu = nn.ReLU()

        self.conv_2 = nn.Conv2d(features_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.unshuffle(x)

        outs = []
        for i, block in enumerate(self.blocks):
            outs.append(block(x[:, i::self.down_reduction]))

        x = torch.cat(outs, dim=1)
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)

        x = self.conv_2(x)

        return x


class DotModel(torch.nn.Module):
    def __init__(self,
                 encoder_name,
                 classes,
                 image_size: tuple[int, int],
                 mask_size: tuple[int, int],
                 reduce_spatial_mode='interpolate',
                 use_dysample=False,
                 use_dynamic_conv=False,
                 num_experts=4,
                 dysample_style='lp',
                 dysample_groups=4,
                 dysample_dyscope=False,
                 use_csnorm=False,  # 新增：是否使用CSNorm
                 csnorm_positions=None,  # 新增：CSNorm插入位置
                 use_pd_attention=False,  # 新增：是否使用基于PD的注意力
                 pd_attention_mode='multiply',  # 新增：注意力应用模式
                 dense_thresh=0.3,  # 新增：密集区域阈值
                 sparse_thresh=0.01,  # 新增：稀疏区域阈值
                 dense_range=(1.2, 1.6),  # 新增：密集区域软标签范围
                 sparse_range=(0.8, 1.2),  # 新增：稀疏区域软标签范围
                 background_range=(0.0, 0.2),  # 新增：背景区域软标签范围
                 ) -> None:
        super().__init__()

        encoder_weights = 'imagenet'
        encoder_depth = 5
        in_channels = 3
        decoder_use_batchnorm: bool = True
        decoder_channels = (256, 128, 64, 32, 16)
        decoder_attention_type = None
        classes = classes
        down_scale = int(image_size[0] / mask_size[0])
        self.use_csnorm = use_csnorm
        self.csnorm_positions = csnorm_positions or ['pre_encoder', 'post_encoder', 'post_encoder_second']
        self.use_pd_attention = use_pd_attention
        self.pd_attention_mode = pd_attention_mode

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        if self.use_csnorm:
            if 'post_encoder_second' in self.csnorm_positions:
                second_ch = self.encoder.out_channels[2]
                if second_ch > 0:
                    self.csnorm_second = CSNorm(second_ch, name="post_encoder_second")
                else:
                    self.csnorm_second = None

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            attention_type=decoder_attention_type,
            use_dysample=use_dysample,
            use_dynamic_conv=use_dynamic_conv,
            num_experts=num_experts,
            dysample_style=dysample_style,
            dysample_groups=dysample_groups,
            dysample_dyscope=dysample_dyscope
        )

        self.segmentation_head = nn.ModuleList()
        for channel in decoder_channels[::-1]:
            self.segmentation_head.append(
                SegmentationHead(
                    in_channels=channel,
                    out_channels=classes,
                    activation=None,
                    kernel_size=3,
                )
            )

        if reduce_spatial_mode == 'interpolate':
            self.pre = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=1. / down_scale, mode='bilinear', align_corners=True),
            )

        elif reduce_spatial_mode == 'pixel':
            self.pre = PixelDistill(in_channels, in_channels, features_channels=32, down_scale=down_scale)
        elif reduce_spatial_mode == 'none':
            self.pre = torch.nn.Identity()
        else:
            raise ValueError('Unknown reduce_spatial_mode')

        if self.use_pd_attention:
            # PD模块的输出通道数 = 输入通道数 = 3
            # 因为PixelDistill(in_channels, in_channels, ...) 输出通道数 = 第二个参数 = in_channels
            pd_out_channels = in_channels  # in_channels = 3

            self.pd_attention = PDAttentionModule(
                in_channels=pd_out_channels,  # 应该是3
                feat_channels=32,
                dense_thresh=dense_thresh,
                sparse_thresh=sparse_thresh,
                dense_range=dense_range,
                sparse_range=sparse_range,
                background_range=background_range
            )

    def forward(self, x, gt_density=None):
        x = self.pre(x)
        # 应用PD注意力
        attention_loss = None
        if self.use_pd_attention:
            # 生成注意力图 (B, 1, H', W')
            attention_map, attention_loss = self.pd_attention(x, gt_density)

            # # 存储损失, attention_loss
            # self.pd_attention_loss = attention_loss
            # 注意力应用：注意力图作用于PD输出
            # attention_map形状: (B, 1, H', W')，需要扩展通道
            attention_map_expanded = attention_map.expand_as(x)  # (B, 3, H', W')

            # 注意力增强：pd_output * attention_map_expanded
            attended_pd_output = x * attention_map_expanded

            # 送入encoder
            x = attended_pd_output
        else:
            x = x

        # 如果没有注意力损失，设为0
        if attention_loss is None:
            attention_loss = torch.tensor(0.0, device=x.device)
        # === 数据流检查点4: encoder输入 ===
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"❌ 错误: encoder输入包含NaN/Inf! min={x.min()}, max={x.max()}")
            # 紧急修复并记录
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            print(f"修复后: min={x.min()}, max={x.max()}")

        features = self.encoder(x)
        if hasattr(self, 'csnorm_second') and self.csnorm_second is not None:
            # 注意：features 是 tuple，需要转换为 list 才能修改
            feat_list = list(features)
            if feat_list[2].size(1) > 0:
                feat_list[2] = self.csnorm_second(feat_list[2])
            features = tuple(feat_list)
        decoder_output = self.decoder(*features)

        masks = []
        for i, seg_head in enumerate(self.segmentation_head):
            masks.append(seg_head(decoder_output[i]))

        return masks[:3], attention_loss


if __name__ == '__main__':
    model = DotModel('mit_b2', 1, (1920, 1088), (960, 544), 'pixel')

    x = torch.randn(1, 3, 1088, 1920)

    y = model(x)
    print(y[0].shape)
