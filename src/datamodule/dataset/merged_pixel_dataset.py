import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List

from src.datamodule.dataset.generic_dataset import GenericDataset


class MergedPixelDataset(GenericDataset):
    def __init__(self, root_data_path: Path, images_list: List[str], image_subdir: str,
                 image_size: Tuple[int, int], mask_size: Tuple[int, int],
                 mosaic: float, transforms=None, normalize=None, is_test=False):
        super().__init__(root_data_path, image_subdir,
                         image_size, mask_size, mosaic, transforms, normalize, is_test)
        self.num_classes = 1
        self.images_list = images_list
        self.mask_generate = self.create_mask_generator(self.num_classes)

    def __len__(self):
        return len(self.images_list)

    def _load_data(self, index: int):
        img_path = str(self.images_list[index])
        # 构造标签路径：将 images 替换为 ground_truth，扩展名改为 .txt
        txt_path = img_path.replace('images', 'ground_truth').replace('.jpg', '.txt')

        frame = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        # 读取像素坐标 .txt（每行 x y）
        try:
            points = np.loadtxt(txt_path, dtype=np.int64).reshape(-1, 2)
        except:
            points = np.zeros((0, 2), dtype=np.int64)

        # 确保坐标在图像范围内（防止异常数据）
        points[:, 0] = np.clip(points[:, 0], 0, w - 1)
        points[:, 1] = np.clip(points[:, 1], 0, h - 1)

        # 组装成模型需要的格式：class, x, y, width, height
        dest_labels = np.zeros((points.shape[0], 5), dtype=np.float32)
        dest_labels[:, 0] = 0          # 单类别，若多类别可提取 data[:,0]
        dest_labels[:, 1:3] = points
        dest_labels[:, 3:5] = 1        # 宽高占位

        # alt 无意义，设为0
        return frame, dest_labels.astype(np.int32), np.array([0.], dtype=np.float32)