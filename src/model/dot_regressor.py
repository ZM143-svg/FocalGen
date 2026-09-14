from typing import Optional

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional

from src.metric.detection_f1 import DetectionF1Metric
from src.losses.dot_loss import DotLoss
from src.metric.counting import CountingMetric
from src.model.dot_model import DotModel
from src.model.lightness_perturbation import LightnessPerturbation
from src.model.CSNorm import CSNorm


class DotRegressor(pl.LightningModule):
    def __init__(self,
                 encoder_name: str,
                 input_channels: int,
                 output_channels: int,
                 spatial_mode: str,
                 loss_function: str,
                 lr: float,
                 train_steps: int,
                 visualize_test_images: bool,
                 obj_threshold: float,
                 image_size: tuple[int, int],
                 mask_size: tuple[int, int],
                 # 调度器参数
                 scheduler_T_0: int = None,
                 scheduler_T_mult: int = 2,
                 scheduler_eta_min: float = 1e-6,
                 use_dysample=False,
                 use_dynamic_conv=False,
                 use_csnorm: bool = False,  # 新增参数
                 csnorm_positions: list = None,  # 新增参数
                 csnorm_training_mode: str = 'freeze_all',  # 新增：训练模式
                 use_pd_attention: bool = False,
                 pd_attention_mode: str = 'multiply',
                 pd_attention_dense_thresh: float = 0.3,
                 pd_attention_sparse_thresh: float = 0.01,
                 pd_attention_dense_range: tuple = (1.2, 1.6),
                 pd_attention_sparse_range: tuple = (0.8, 1.2),
                 pd_attention_background_range: tuple = (0.0, 0.2),
                 pd_attention_loss_weight: float = 1.0,
                 # pd_attention_training_phase: str = 'joint',
                 pd_attention_only_epochs: int = 6,
                 freeze_main_net_epochs: int = 8,
                 ):
        super().__init__()

        # params
        self._encoder_name = encoder_name
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._spatial_mode = spatial_mode
        self._loss_function = loss_function
        self._lr = lr
        self._train_steps = train_steps
        self._visualize_test_images = visualize_test_images
        self._obj_threshold = obj_threshold
        self._image_size = image_size
        self._mask_size = mask_size
        self.scheduler_T_0 = scheduler_T_0
        self.scheduler_T_mult = scheduler_T_mult
        self.scheduler_eta_min = scheduler_eta_min
        # 新增CSNorm参数
        self.use_csnorm = use_csnorm
        self.csnorm_positions = csnorm_positions or ['pre_encoder', 'post_encoder', 'post_encoder_second']
        self.csnorm_training_mode = csnorm_training_mode
        # params
        # 保存PD注意力参数
        self.use_pd_attention = use_pd_attention
        self.pd_attention_mode = pd_attention_mode
        self.pd_attention_dense_thresh = pd_attention_dense_thresh
        self.pd_attention_sparse_thresh = pd_attention_sparse_thresh
        self.pd_attention_dense_range = pd_attention_dense_range
        self.pd_attention_sparse_range = pd_attention_sparse_range
        self.pd_attention_background_range = pd_attention_background_range
        self.pd_attention_loss_weight = pd_attention_loss_weight
        # self.pd_attention_training_phase = pd_attention_training_phase
        self.pd_attention_only_epochs = pd_attention_only_epochs
        self.freeze_main_net_epochs = freeze_main_net_epochs

        # network
        self.network = DotModel(
            encoder_name=self._encoder_name,
            classes=self._output_channels,
            reduce_spatial_mode=self._spatial_mode,
            image_size=self._image_size,
            mask_size=self._mask_size,
            use_dysample=use_dysample,
            use_dynamic_conv=use_dynamic_conv,
            use_csnorm=self.use_csnorm,
            csnorm_positions=self.csnorm_positions,
            # 添加PD注意力参数
            use_pd_attention=self.use_pd_attention,
            pd_attention_mode=self.pd_attention_mode,
            dense_thresh=self.pd_attention_dense_thresh,
            sparse_thresh=self.pd_attention_sparse_thresh,
            dense_range=self.pd_attention_dense_range,
            sparse_range=self.pd_attention_sparse_range,
            background_range=self.pd_attention_background_range,
        )

        # loss
        if self._loss_function == 'dot':
            self.loss = DotLoss()
        elif self._loss_function == 'mse':
            self.loss = torch.nn.MSELoss()
        else:
            raise NotImplementedError(f'Unsupported loss function: {self._loss_function}')

        # metrics
        self.val_f1 = DetectionF1Metric(correct_distance=5.0)
        self.test_f1 = DetectionF1Metric(correct_distance=5.0)

        self.val_count = CountingMetric(self._output_channels)
        self.test_count = CountingMetric(self._output_channels)

        self.save_hyperparameters()

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def forward(self, x: torch.Tensor, gt_density: torch.Tensor = None) -> torch.Tensor:
        # 获取预测和注意力损失
        masks, attention_loss = self.network(x, gt_density)

        out, out_x2, out_x4 = masks

        out = nn.functional.interpolate(out, size=(self._mask_size[1], self._mask_size[0]), mode='bilinear',
                                        align_corners=True)
        out_x2 = nn.functional.interpolate(out_x2, size=(self._mask_size[1], self._mask_size[0]), mode='bilinear',
                                           align_corners=True)
        out_x4 = nn.functional.interpolate(out_x4, size=(self._mask_size[1], self._mask_size[0]), mode='bilinear',
                                           align_corners=True)

        return (out, out_x2, out_x4), attention_loss

    def calculate_loss(self, y_pred, mask, gt, is_stage=False):
        if self._loss_function == 'dot':
            return self.loss(y_pred, self.postprocessing(y_pred, thresh=min(0.1 + (self.current_epoch / 25),
                                                                            self._obj_threshold)), mask, gt,
                             is_stage=is_stage)
        elif self._loss_function == 'mse':
            return self.loss(y_pred, mask), {}
        else:
            raise NotImplementedError(f'Unsupported loss function: {self._loss_function}')

    def _reset_csnorm_counters(self):
        """重置所有CSNorm模块的计数器"""
        for module in self.network.modules():
            if isinstance(module, CSNorm):
                module.step_counter = 0
        # 重置网络主计数器
        if hasattr(self.network, 'step_counter'):
            self.network.step_counter = 0

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Optional[torch.Tensor]:
        if batch_idx == 0 and self.use_csnorm:
            self._reset_csnorm_counters()

        image, mask, gt, alt = batch

        current_epoch = self.current_epoch
        # current_stage = self._get_current_training_stage(current_epoch)
        # # 只在每个epoch的第一个batch打印阶段信息
        # if batch_idx == 0:
        #     print(f"\nEpoch {current_epoch} - 阶段: {current_stage}")
        if current_epoch < 10:
            perturb_prob = 0.5  # 初期使用较低概率
        elif current_epoch < 30:
            perturb_prob = 0.7  # 中期提高概率
        else:
            perturb_prob = 0.4  # 后期稍微降低，专注于微调
        # # 应用光照扰动（仅在CSNorm训练模式下）
        # if self.use_csnorm:
        #     self.perturb_generator = LightnessPerturbation(perturb_prob)
        #     image = self.perturb_generator(image)

        (out, out_x2, out_x4), attention_loss = self.forward(image, mask)

        if self._loss_function == 'mse':
            mask *= 500

        loss_x2, loss_d = self.calculate_loss(out, mask, gt)
        loss_x4, _ = self.calculate_loss(out_x2, mask, gt, is_stage=True)
        loss_x8, _ = self.calculate_loss(out_x4, mask, gt, is_stage=True)

        main_loss = loss_x2 + loss_x4 * 0.7 + loss_x8 * 0.3

        # # 计算注意力损失（如果使用PD注意力）
        # attention_loss = torch.tensor(0.0, device=image.device)
        if self.use_pd_attention:
            # attention_loss = getattr(self.network, 'pd_attention_loss', torch.tensor(0.0))
            # 联合训练：主损失 + 注意力监督损失
            # （分阶段训练策略已停用，相关代码见下方注释）
            loss = main_loss + attention_loss * 0.1

            # 记录注意力损失
            self.log('train_attention_loss', attention_loss, on_step=True, on_epoch=True, sync_dist=True)
        else:
            loss = main_loss

        # ===== 以下为已停用的分阶段训练策略（current_stage 未定义，故不再使用）=====
        # if current_stage == 'attention_only':
        #     loss = attention_loss * 1.0
        # elif current_stage == 'freeze_main':
        #     loss = attention_loss * self.pd_attention_loss_weight + main_loss * 0.2
        # elif current_stage == 'unfreeze_decoder':
        #     loss = attention_loss * 0.2 + main_loss * 0.8
        # else:
        #     loss = main_loss + attention_loss * 0.1

        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_main_loss', main_loss, on_step=True, on_epoch=True, sync_dist=True)

        for key, value in loss_d.items():
            self.log(f'train_{key}', value, on_step=True, on_epoch=True, sync_dist=True)

        return loss

        # self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)

        # for key, value in loss_d.items():
        #     self.log(f'train_{key}', value, on_step=True, on_epoch=True, sync_dist=True)

        # return loss

    # def on_train_epoch_start(self):
    #     """在每个训练epoch开始时调用，确保优化器配置正确"""
    #     super().on_train_epoch_start()

    #     # 重新配置优化器（如果阶段变化）
    #     current_stage = self._get_current_training_stage(self.current_epoch)

    #     # 检查是否需要重新配置优化器
    #     if not hasattr(self, '_last_stage') or current_stage != self._last_stage:
    #         self._last_stage = current_stage
    #         print(f"\n=== 阶段切换: {current_stage} ===")
    #         print(f"重新配置优化器...")

    #         # 重新设置参数冻结
    #         self._setup_training_stage(current_stage)

    #         # 重新配置优化器
    #         self._reconfigure_optimizer(current_stage)

    # def _reconfigure_optimizer(self, current_stage):
    #     """重新配置优化器"""
    #     # 收集可训练参数
    #     trainable_params = []
    #     for name, param in self.named_parameters():
    #         if param.requires_grad:
    #             trainable_params.append(param)

    #     print(f"可训练参数数量: {len(trainable_params)}")

    #     # 创建新的优化器
    #     if current_stage == 'attention_only':
    #         lr = self._lr * 0.1
    #     elif current_stage == 'freeze_main':
    #         lr = self._lr * 0.8
    #     elif current_stage == 'unfreeze_decoder':
    #         lr = self._lr * 0.3  # 4e-5
    #     elif current_stage == 'joint':
    #         lr = self._lr * 0.05  # 8e-6，更小的学习率

    #     new_optimizer = torch.optim.AdamW(
    #         trainable_params,
    #         lr=lr,
    #         weight_decay=1e-4,
    #     )

    #     # 更新训练器的优化器
    #     if hasattr(self, 'trainer') and self.trainer is not None:
    #         self.trainer.optimizers = [new_optimizer]
    #         print(f"优化器已更新，学习率: {lr}")

    #     # 更新内部的优化器引用
    #     self._optimizer = new_optimizer

    #
    # def _get_current_training_stage(self, current_epoch):
    #     """根据当前epoch确定训练阶段"""
    #     if not self.use_pd_attention:
    #         return 'full'

    #     if current_epoch < self.pd_attention_only_epochs:
    #         return 'attention_only'
    #     elif current_epoch < self.pd_attention_only_epochs + self.freeze_main_net_epochs:
    #         return 'freeze_main'
    #     elif current_epoch < self.pd_attention_only_epochs + self.freeze_main_net_epochs + 8:
    #         return 'unfreeze_decoder'
    #     else:
    #         return 'joint'

    # def _setup_training_stage(self, current_stage):
    #     """根据训练阶段设置参数冻结"""
    #     if not self.use_pd_attention:
    #         return

    #     if current_stage == 'attention_only':
    #         # 只训练注意力模块，冻结其他所有参数
    #         for name, param in self.network.named_parameters():
    #             if 'pd_attention' in name:
    #                 param.requires_grad = True
    #             else:
    #                 param.requires_grad = False

    #     elif current_stage == 'freeze_main':
    #         # 冻结主网络，训练注意力模块
    #         for name, param in self.network.named_parameters():
    #             if 'encoder' in name or 'decoder' in name or 'segmentation_head' in name:
    #                 param.requires_grad = False
    #             else:
    #                 param.requires_grad = True

    #     elif current_stage == 'unfreeze_decoder':
    #         # 解冻encoder，冻结decoder
    #         for name, param in self.network.named_parameters():
    #             if 'encoder' in name:
    #                 param.requires_grad = False
    #             elif 'decoder' in name or 'segmentation_head' in name:
    #                 param.requires_grad = True
    #             else:
    #                 param.requires_grad = True  # 注意力模块继续训练

        # elif current_stage == 'joint':
        #     # 联合训练，解冻所有参数
        #     for param in self.network.parameters():
        #         param.requires_grad = True

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        image, mask, gt, alt = batch

        (y_pred, _, _), _ = self.forward(image, None)

        assert y_pred.shape == mask.shape, f'Predicted shape: {y_pred.shape}, mask shape: {mask.shape}'

        predicted_points = self.postprocessing(y_pred)

        f1, precission, recall = self.val_f1(predicted_points, gt)
        self.log('val_f1', f1, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_P', precission, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_R', recall, on_step=False, on_epoch=True, sync_dist=True)

        mae, mae_norm = self.val_count(predicted_points, gt)
        self.log('val_mae', mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_mae_norm', mae_norm, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        image, mask, gt, alt, _, _ = batch

        (y_pred, _, _), _ = self.forward(image, None)

        predicted_points = self.postprocessing(y_pred)

        f1, precission, recall = self.test_f1(predicted_points, gt)
        self.log('test_f1', f1, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_P', precission, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_R', recall, on_step=False, on_epoch=True, sync_dist=True)

        mae, mae_norm = self.test_count(predicted_points, gt)
        self.log('test_mae', mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_mae_norm', mae_norm, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):

        self._setup_parameter_freezing()
        # 获取当前训练阶段（基于当前epoch）
        # current_stage = self._get_current_training_stage(self.current_epoch)

        # print(f"\n=== 训练阶段配置 ===")
        # print(f"当前epoch: {self.current_epoch}")
        # print(f"当前阶段: {current_stage}")
        # self._setup_training_stage(current_stage)

        # 收集可训练参数并打印详细信息
        trainable_params = []
        trainable_names = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                trainable_names.append(name)

        print(f"可训练参数数量: {len(trainable_params)} / {len(list(self.parameters()))}")
        print(f"可训练参数详情:")
        for i, name in enumerate(trainable_names[:10]):  # 只显示前10个，避免日志过长
            print(f"  {i + 1}. {name}")
        if len(trainable_names) > 10:
            print(f"  ... 还有 {len(trainable_names) - 10} 个可训练参数")

        # # 根据训练阶段调整学习率
        # if current_stage == 'attention_only':
        #     lr = self._lr * 0.1
        #     print(f"注意力模块专项训练，使用学习率: {lr}")
        # elif current_stage == 'freeze_main':
        #     lr = self._lr * 0.8
        #     print(f"冻结主网络阶段，使用中等学习率: {lr}")

        # elif current_stage == 'unfreeze_decoder':
        #     lr = self._lr * 0.3
        #     print(f"解冻一部分，使用学习率：{lr}")
        # else:
        #     lr = self._lr * 0.05
        #     print(f"联合训练阶段，使用基础学习率: {lr}")
        

        if self.use_csnorm:
            lr = self._lr * 0.1  # 降低学习率
            print(f"CSNorm专项训练，使用降低的学习率: {lr}")
        else:
            lr = self._lr

        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

        if self.scheduler_T_0 is not None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.scheduler_T_0,
                T_mult=self.scheduler_T_mult,
                eta_min=self.scheduler_eta_min
            )

            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=1000
            )

            combined_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[1000]
            )

            schedule = {
                'scheduler': combined_scheduler,
                'interval': 'step',
            }
            return [optimizer], [schedule]
        else:
            return [optimizer]

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=self._train_steps // 2,
        # )

        # schedule = {
        #     'scheduler': scheduler,
        #     'interval': 'step',
        # }
        # return [optimizer], [schedule]

    def _setup_parameter_freezing(self):
        """根据训练模式设置参数冻结"""
        if not self.use_csnorm:
            return

        if self.csnorm_training_mode == 'freeze_all':
            # 冻结所有非CSNorm参数，只训练CSNorm
            print("=== 冻结策略: 训练CSNorm模块，冻结非CSNorm模块 ===")

            # 首先解冻所有参数
            for param in self.network.parameters():
                param.requires_grad = True

            # 然后冻结非CSNorm参数
            for name, param in self.network.named_parameters():
                if 'csnorm_second' in name:
                    param.requires_grad = True
                    print(f"训练CSNorm参数: {name}")
                else:
                    param.requires_grad = False
                    print(f"冻结非CSNorm参数: {name}")

    def postprocessing(self, y_pred_raw: torch.Tensor, thresh: float = None) -> torch.Tensor:
        if self._loss_function == 'mse':
            y_pred = y_pred_raw / 500.
        else:
            y_pred = torch.sigmoid(y_pred_raw)

        y_pred = self._nms(y_pred)

        return_values = []

        thresh = self._obj_threshold if thresh is None else thresh

        for batch_id in range(y_pred.shape[0]):
            pred_b = y_pred[batch_id]

            predictions = []

            for class_id in range(pred_b.shape[0]):
                yx = torch.argwhere(pred_b[class_id] > thresh)

                if yx.shape[0] > 0:
                    predictions.append(
                        torch.cat([
                            torch.full((yx.shape[0], 1), class_id, dtype=torch.int32).to(
                                y_pred.device),
                            yx[:, [1, 0]],
                            y_pred_raw[batch_id][class_id][yx[:, 0],
                            yx[:, 1]][:, None],
                        ], 1),
                    )

            if len(predictions) > 0:
                predictions = torch.cat(predictions, dim=0)
            else:
                predictions = torch.zeros(
                    (0, 4), dtype=torch.float32).to(y_pred.device)

            predictions[:, 1] *= self._image_size[0] / self._mask_size[0]
            predictions[:, 2] *= self._image_size[1] / self._mask_size[1]

            return_values.append(predictions)

        return return_values

    @staticmethod
    def _nms(heat, kernel=3):
        pad = (kernel - 1) // 2

        hmax = torch.nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()

        return heat * keep

    # def load_state_dict(self, state_dict, strict=True):
    #     """重写load_state_dict方法以处理PD注意力和CSNorm权重不匹配"""
    #     if self.use_csnorm or self.use_pd_attention:
    #         print("=== 使用非严格模式加载权重，处理新添加的模块 ===")

    #         # 获取当前模型的状态字典
    #         model_state_dict = self.state_dict()
    #         filtered_state_dict = {}

    #         # 要跳过的模块列表（新添加的，检查点中可能没有）
    #         skip_prefixes = []

    #         if self.use_pd_attention:
    #             skip_prefixes.append('network.pd_attention.')
    #             print("注意：跳过PD注意力模块权重，将使用随机初始化")

    #         # if self.use_csnorm:
    #         #     skip_prefixes.extend(['network.csnorm_modules.', 'csnorm_modules.'])
    #         #     print("注意：跳过CSNorm模块权重，将使用随机初始化")

    #         # 过滤权重
    #         for k, v in state_dict.items():
    #             # 检查是否应该跳过这个权重
    #             should_skip = any(k.startswith(prefix) for prefix in skip_prefixes)

    #             if should_skip:
    #                 print(f"跳过权重: {k} (新添加模块)")
    #                 continue

    #             if k in model_state_dict and model_state_dict[k].shape == v.shape:
    #                 filtered_state_dict[k] = v
    #             elif not self.training:  # 仅在训练时严格检查
    #                 print(f"警告: 权重形状不匹配或找不到: {k}")
    #             else:
    #                 # 训练时可以跳过不匹配的权重
    #                 pass

    #         # 更新模型状态字典
    #         model_state_dict.update(filtered_state_dict)

    #         # 使用非严格模式加载
    #         return super().load_state_dict(model_state_dict, strict=False)
    #     else:
    #         return super().load_state_dict(state_dict, strict=strict)

    # @classmethod
    # def load_from_checkpoint(cls, checkpoint_path, **kwargs):
    #     """重写load_from_checkpoint以处理CSNorm、PD注意力和PyTorch 2.6兼容性"""
    #     # 先创建模型实例
    #     model = cls(**kwargs)

    #     # 关键修改：添加weights_only=False以兼容PyTorch 2.6
    #     try:
    #         checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    #     except Exception as e:
    #         print(f"使用weights_only=False加载失败: {e}")
    #         # 回退方案：尝试使用旧版加载方式
    #         checkpoint = torch.load(checkpoint_path, map_location='cpu')

    #     # 加载状态字典，使用非严格模式
    #     state_dict = checkpoint.get('state_dict', checkpoint)

    #     # 手动处理权重加载，允许跳过不匹配的权重
    #     model_state_dict = model.state_dict()
    #     filtered_state_dict = {}

    #     # 确定要跳过的模块前缀
    #     skip_prefixes = []
    #     if kwargs.get('use_pd_attention', False):
    #         skip_prefixes.append('network.pd_attention.')
    #         print("PD注意力模块将随机初始化")

    #     # if kwargs.get('use_csnorm', False):
    #     #     skip_prefixes.extend(['network.csnorm_modules.', 'csnorm_modules.'])
    #     #     print("CSNorm模块将随机初始化")

    #     loaded_count = 0
    #     skipped_count = 0

    #     for k, v in state_dict.items():
    #         # 检查是否应该跳过
    #         should_skip = any(k.startswith(prefix) for prefix in skip_prefixes)

    #         if should_skip:
    #             print(f"跳过: {k}")
    #             skipped_count += 1
    #             continue

    #         if k in model_state_dict and model_state_dict[k].shape == v.shape:
    #             filtered_state_dict[k] = v
    #             loaded_count += 1
    #         else:
    #             print(f"跳过不匹配的权重: {k}")
    #             skipped_count += 1

    #     # 更新模型状态字典
    #     model_state_dict.update(filtered_state_dict)
    #     model.load_state_dict(model_state_dict, strict=False)

    #     print(f"成功加载 {loaded_count} 个权重参数，跳过 {skipped_count} 个")

    #     return model

    def load_state_dict(self, state_dict, strict=True):
        """重写load_state_dict方法以处理CSNorm权重不匹配"""
        if self.use_csnorm:
            # 过滤掉CSNorm相关的权重（如果检查点中没有）
            model_state_dict = self.state_dict()
            filtered_state_dict = {}
    
            for k, v in state_dict.items():
                if k in model_state_dict and model_state_dict[k].shape == v.shape:
                    filtered_state_dict[k] = v
                elif 'csnorm_modules' in k:
                    print(f"跳过CSNorm权重: {k}")
                else:
                    print(f"跳过不匹配的权重: {k}")
    
            # 使用非严格模式加载
            return super().load_state_dict(filtered_state_dict, strict=False)
        else:
            return super().load_state_dict(state_dict, strict=strict)
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        """重写load_from_checkpoint以处理CSNorm和PyTorch 2.6兼容性"""
        # 先创建模型实例
        model = cls(**kwargs)
    
        # 关键修改：添加weights_only=False以兼容PyTorch 2.6
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"使用weights_only=False加载失败: {e}")
            # 回退方案：尝试使用旧版加载方式
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
        # 过滤状态字典
        if model.use_csnorm:
            model_state_dict = model.state_dict()
            filtered_state_dict = {}
    
            for k, v in checkpoint['state_dict'].items():
                if k in model_state_dict and model_state_dict[k].shape == v.shape:
                    filtered_state_dict[k] = v
                elif 'csnorm_modules' in k:
                    print(f"跳过CSNorm权重: {k}")
                else:
                    print(f"跳过不匹配的权重: {k}")
    
            # 更新模型状态字典
            model_state_dict.update(filtered_state_dict)
            model.load_state_dict(model_state_dict, strict=False)
    
            print(f"成功加载 {len(filtered_state_dict)} 个权重参数")
        else:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
    
        return model

