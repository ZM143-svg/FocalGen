# lightness_perturbation.py
import torch
import torch.fft
import random


class LightnessPerturbation:
    """基于频率域的lightness扰动 - 按照CSNorm论文实现"""

    def __init__(self, apply_probability=0.8):
        self.apply_prob = apply_probability

    def __call__(self, images):
        """
        输入: batch of images [B, C, H, W], 值范围[0,1]
        输出: 频率域扰动后的batch
        """
        batch_size = images.shape[0]
        perturbed_images = []

        for i in range(batch_size):
            img = images[i].clone()

            # 随机决定是否应用扰动
            if random.random() > self.apply_prob:
                perturbed_images.append(img)
                continue

            try:
                # 将图像转换到频率域
                img_fft = torch.fft.fft2(img, dim=(-2, -1))
                img_fft_shift = torch.fft.fftshift(img_fft)

                # 获取幅度(amplitude)和相位(phase)
                amplitude = torch.abs(img_fft_shift)
                phase = torch.angle(img_fft_shift)

                # 按照论文方法对幅度进行扰动
                # 使用随机插值系数，模拟不同lightness条件
                lambda_val = random.uniform(0.3, 0.7)  # 论文中的λ在[0,1]范围内

                # 对幅度进行线性插值扰动
                # 这里简化实现：对当前幅度进行缩放来模拟lightness变化
                scale_factor = random.choice([0.5, 0.7, 1.0, 1.3, 1.5, 2.0])  # 模拟不同lightness
                perturbed_amplitude = amplitude * scale_factor

                # 重建频率域表示
                perturbed_fft_shift = perturbed_amplitude * torch.exp(1j * phase)
                perturbed_fft = torch.fft.ifftshift(perturbed_fft_shift)
                perturbed_img = torch.fft.ifft2(perturbed_fft, dim=(-2, -1))

                # 取实部并确保值范围合理
                perturbed_img = torch.real(perturbed_img)
                perturbed_img = torch.clamp(perturbed_img, 0, 1)

                perturbed_images.append(perturbed_img)

            except Exception as e:
                print(f"频率域扰动错误: {e}，使用原始图像")
                perturbed_images.append(img)

        return torch.stack(perturbed_images)