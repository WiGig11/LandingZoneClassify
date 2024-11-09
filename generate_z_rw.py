import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
"""
generate random runways
"""
outdir = './Landing_dataset/Z'
# 定义图像尺寸
width, height = 100,100

import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image

for i in range(600,800):
    # 生成随机噪声，并应用高斯滤波
    noise = np.random.rand(height, width)
    noise = gaussian_filter(noise, sigma=2)

    # 定义三种颜色
    color1 = np.array([0xEA, 0x90, 0x68]) / 255  # 浅色
    color2 = np.array([0x81, 0x43, 0x30]) / 255  # 中间色
    color3 = np.array([0x3D, 0x20, 0x1A]) / 255  # 深色

    # 创建一个空的纹理数组
    texture = np.zeros((height, width, 3))

    # 分段插值，调整浅色区域的比例
    mask1 = noise < 0.5
    texture[mask1] = np.outer(1 - noise[mask1] / 0.5, color1) + np.outer(noise[mask1] / 0.5, color2)

    mask2 = (noise >= 0.5) & (noise < 0.75)
    texture[mask2] = np.outer(1 - (noise[mask2] - 0.5) / 0.25, color2) + np.outer((noise[mask2] - 0.5) / 0.25, color3)

    mask3 = noise >= 0.75
    texture[mask3] = color3

    # 限制颜色值在 [0,1] 范围内
    texture = np.clip(texture, 0, 1)

    # 将颜色值转换为 0-255 范围的整数，并创建Pillow图像
    texture_image = (texture * 255).astype(np.uint8)
    image = Image.fromarray(texture_image)

    # 保存图像为 PNG 文件
    path = os.path.join(outdir,f'{i}.png')
    image.save(path)
    print("Image saved as texture_image.png")
