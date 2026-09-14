import os

os.environ['OMP_NUM_THREADS'] = '4'

import hydra
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.distributed.algorithms.ddp_comm_hooks import (
    default_hooks as default,
)

from src.model.dot_regressor import DotRegressor
from src.datamodule.dot_datamodule import DotDatamodule


@hydra.main(version_base=None, config_path='./configs/')
def main(cfg: DictConfig) -> None:
    pl.seed_everything(seed=42)

    datamodule = DotDatamodule(
        root_data_path=cfg.data_path,
        dataset=cfg.dataset,
        data_fold=cfg.data_fold,
        batch_size=cfg.batch_size,
        workers=cfg.workers,
        data_mean=cfg.data_mean,
        data_std=cfg.data_std,
        image_size=cfg.image_size,
        mask_size=cfg.mask_size,
        mosaic=cfg.mosaic,
    )

    if cfg.restore_from_ckpt is not None:
        print(f"从 checkpoint 加载模型权重（不恢复优化器）: {cfg.restore_from_ckpt}")
        # 使用你自定义的 load_from_checkpoint 加载权重
        model = DotRegressor.load_from_checkpoint(
            cfg.restore_from_ckpt,
            # 必须传入所有构造函数参数，与当前模型一致
            encoder_name=cfg.encoder_name,
            input_channels=cfg.input_channels,
            output_channels=cfg.output_channels,
            spatial_mode=cfg.spatial_mode,
            loss_function=cfg.loss,
            lr=cfg.lr,
            train_steps=datamodule.get_train_steps(),
            visualize_test_images=cfg.visualize_test_images if not cfg.debug else False,
            obj_threshold=cfg.obj_threshold,
            image_size=cfg.image_size,
            mask_size=cfg.mask_size,
            scheduler_T_0=cfg.get('scheduler_T_0', datamodule.get_train_steps() // 4),
            scheduler_T_mult=cfg.get('scheduler_T_mult', 2),
            scheduler_eta_min=cfg.get('scheduler_eta_min', 1e-6),
            use_dysample=cfg.get('use_dysample', False),
            use_dynamic_conv=cfg.get('use_dynamic_conv', False),
            use_csnorm=cfg.get('use_csnorm', False),
            csnorm_positions=cfg.get('csnorm_positions', ['pre', 'post_encoder', 'post_encoder_second']),
            csnorm_training_mode=cfg.get('csnorm_training_mode', 'freeze_all'),
            # PD注意力参数
            use_pd_attention=cfg.get('use_pd_attention', False),
            pd_attention_mode=cfg.get('pd_attention_mode', 'multiply'),
            pd_attention_dense_thresh=cfg.get('pd_attention_dense_thresh', 0.3),
            pd_attention_sparse_thresh=cfg.get('pd_attention_sparse_thresh', 0.01),
            pd_attention_dense_range=tuple(cfg.get('pd_attention_dense_range', [1.2, 1.6])),
            pd_attention_sparse_range=tuple(cfg.get('pd_attention_sparse_range', [0.8, 1.2])),
            pd_attention_background_range=tuple(cfg.get('pd_attention_background_range', [0.0, 0.2])),
            pd_attention_loss_weight=cfg.get('pd_attention_loss_weight', 0.1),
            # pd_attention_training_phase=cfg.get('pd_attention_training_phase', 'joint'),
            pd_attention_only_epochs=cfg.get('pd_attention_only_epochs', 10),
            freeze_main_net_epochs=cfg.get('freeze_main_net_epochs', 10),
        )
        # 重要：设置 ckpt_path = None，避免 Lightning 再次尝试恢复优化器
        ckpt_path = None
    else:
        print('Creating new model...')
        model = DotRegressor(
            encoder_name=cfg.encoder_name,
            input_channels=cfg.input_channels,
            output_channels=cfg.output_channels,
            spatial_mode=cfg.spatial_mode,
            loss_function=cfg.loss,
            lr=cfg.lr,
            train_steps=datamodule.get_train_steps(),
            visualize_test_images=cfg.visualize_test_images if not cfg.debug else False,
            obj_threshold=cfg.obj_threshold,
            image_size=cfg.image_size,
            mask_size=cfg.mask_size,
            scheduler_T_0=cfg.get('scheduler_T_0', datamodule.get_train_steps() // 4),
            scheduler_T_mult=cfg.get('scheduler_T_mult', 2),
            scheduler_eta_min=cfg.get('scheduler_eta_min', 1e-6),
            use_dysample=cfg.get('use_dysample', False),
            use_dynamic_conv=cfg.get('use_dynamic_conv', False),
            use_csnorm=cfg.get('use_csnorm', False),
            csnorm_positions=cfg.get('csnorm_positions', ['pre', 'post_encoder', 'post_encoder_second']),
            csnorm_training_mode=cfg.get('csnorm_training_mode', 'freeze_all'),
            # PD注意力参数
            use_pd_attention=cfg.get('use_pd_attention', False),
            pd_attention_mode=cfg.get('pd_attention_mode', 'multiply'),
            pd_attention_dense_thresh=cfg.get('pd_attention_dense_thresh', 0.3),
            pd_attention_sparse_thresh=cfg.get('pd_attention_sparse_thresh', 0.01),
            pd_attention_dense_range=tuple(cfg.get('pd_attention_dense_range', [1.2, 1.6])),
            pd_attention_sparse_range=tuple(cfg.get('pd_attention_sparse_range', [0.8, 1.2])),
            pd_attention_background_range=tuple(cfg.get('pd_attention_background_range', [0.0, 0.2])),
            pd_attention_loss_weight=cfg.get('pd_attention_loss_weight', 0.1),
            # pd_attention_training_phase=cfg.get('pd_attention_training_phase', 'joint'),
            pd_attention_only_epochs=cfg.get('pd_attention_only_epochs', 10),
            freeze_main_net_epochs=cfg.get('freeze_main_net_epochs', 10),
        )
            # ckpt_path = None
     # 打印PD注意力配置信息
    if cfg.get('use_pd_attention', False):
        print("=== PD注意力模块配置 ===")
        print(f"注意力损失权重: {cfg.get('pd_attention_loss_weight', 0.1)}")
        print(f"注意力单独训练轮次: {cfg.get('pd_attention_only_epochs', 10)}")
        print(f"冻结主网络轮次: {cfg.get('freeze_main_net_epochs', 10)}")
    # ========== CSNorm训练的特殊设置 ==========
    if cfg.get('use_csnorm', False):
        print("=== CSNorm训练模式 ===")
        print(f"CSNorm位置: {cfg.get('csnorm_positions', ['pre_encoder', 'post_encoder', 'post_encoder_second'])}")
        print(f"训练模式: {cfg.get('csnorm_training_mode', 'freeze_all')}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir'],
        filename='epoch_{epoch}-f1_{val_f1:.2f}',
        monitor=cfg.monitor,
        auto_insert_metric_name=False,
        verbose=True,
        save_last=True,
        mode=cfg.monitor_mode)

    model_summary_callback = ModelSummary(max_depth=1)

    early_stopping_callback = EarlyStopping(
        monitor=cfg.monitor,
        mode=cfg.monitor_mode,
        patience=cfg.es_patience)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # 创建 CSVLogger（总是启用）
    csv_logger = CSVLogger(save_dir=hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir'],
                           name='csv_logs',  # 会创建子目录
                           version=None)  # 自动版本号

    logger = [csv_logger]
    # NeptuneLogger 为可选：仅在非 debug 且提供 NEPTUNE_API_TOKEN 时启用
    if not cfg.debug and os.environ.get('NEPTUNE_API_TOKEN'):
        neptune_logger = NeptuneLogger(
            api_key=os.environ['NEPTUNE_API_TOKEN'],
            project=cfg.get('neptune_project', 'FocalGen/crowd-localization'),
            log_model_checkpoints=False,
        )
        logger.append(neptune_logger)
        print('检测到 NEPTUNE_API_TOKEN，已启用 NeptuneLogger。')
    elif not cfg.debug:
        print('未检测到 NEPTUNE_API_TOKEN，仅使用 CSVLogger 记录日志。')

    # if not cfg.debug:
    #     logger = TensorBoardLogger(
    #         save_dir='./logs/',
    #         name='UP-COUNT',
    #     )
    # else:
    #     logger = None

    callbacks = [
        checkpoint_callback,
        model_summary_callback,
        early_stopping_callback,
    ]

    if not cfg.debug:
        callbacks.append(lr_monitor)

    torch.set_float32_matmul_precision('medium')
    # ========== 关键修改：不传递ckpt_path给trainer.fit ==========
    # 因为我们已经手动加载了权重，避免PyTorch Lightning再次尝试加载
    ckpt_path = None

    # ========== CSNorm训练的特殊训练器设置 ==========
    if cfg.get('use_csnorm', False) and cfg.get('csnorm_training_mode') == 'freeze_all':
        max_epochs = cfg.get('csnorm_epochs', 50)
        print(f"CSNorm专项训练: {max_epochs} 轮")
    else:
        max_epochs = cfg.epochs
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        devices="auto" if cfg.devices <= 0 else cfg.devices,
        accelerator='gpu' if torch.cuda.is_available() and cfg.devices > 0 else 'cpu',
        precision=cfg.precision,
        max_epochs=max_epochs,
        benchmark=True if cfg.devices > 0 else False,
        sync_batchnorm=True if torch.cuda.is_available() else False,
        accumulate_grad_batches=4,
        gradient_clip_val=cfg.get('gradient_clip_val', 0.7),
    )

    if not cfg.test_only:
        # ckpt_path = cfg.restore_from_ckpt if cfg.restore_from_ckpt is not None else None
        trainer.fit(model, datamodule, ckpt_path=ckpt_path)
        trainer.test(model, datamodule, ckpt_path='best')
    else:
        assert cfg.restore_from_ckpt is not None
        trainer.test(model, datamodule, ckpt_path=cfg.restore_from_ckpt)


if __name__ == '__main__':
    main()
