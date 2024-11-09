import numpy as np
import cv2
from scipy.signal import convolve2d
from skimage.restoration import wiener
import matplotlib.pyplot as plt

# 生成运动模糊核 (PSF: Point Spread Function)
def motion_blur_psf(length, angle):
    psf = np.zeros((length, length))
    center = length // 2
    slope = np.tan(np.deg2rad(angle))
    
    for i in range(length):
        offset = int(slope * (i - center))
        psf[center + offset, i] = 1
    
    return psf / psf.sum()

# 模拟图像模糊
def blur_image(image, psf):
    return convolve2d(image, psf, mode='same')

# 维纳滤波去模糊
def wiener_deblur(blurred_image, psf, K=0.01):
    return wiener(blurred_image, psf, K)

# 读取和预处理图像
image = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)  # 加载灰度图像
image = image / 255.0  # 将图像归一化到0-1之间

# 定义模糊核
psf = motion_blur_psf(length=5, angle=3)  # 生成长度为15，角度为30度的运动模糊核

# 模糊图像
blurred_image = blur_image(image, psf)
cv2.imwrite('blurred.png', blurred_image)

# 应用维纳滤波进行去模糊
restored_image = wiener_deblur(blurred_image, psf)

plt.title('Blurred Image')
plt.imshow(blurred_image, cmap='gray')
plt.savefig('blurred.png')

# 显示图像
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Blurred Image')
plt.imshow(blurred_image, cmap='gray')
plt.savefig('blurred.png')

plt.subplot(1, 3, 3)
plt.title('Restored Image')
plt.imshow(restored_image, cmap='gray')

plt.show()

import cv2
import numpy as np

# 判断图像是否模糊
def detect_blur(image, threshold=100):
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 计算拉普拉斯变换并求方差
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 根据方差判断是否模糊
    if laplacian_var < threshold:
        return True, laplacian_var  # 图像模糊
    else:
        return False, laplacian_var  # 图像清晰

# 读取图像
image = cv2.imread('blurred.png')

# 判断图像是否模糊
is_blurred, score = detect_blur(image)

# 输出结果
if is_blurred:
    print(f"Image is blurry. Laplacian variance: {score}")
else:
    print(f"Image is clear. Laplacian variance: {score}")

image = cv2.imread('test.png')

# 判断图像是否模糊
is_blurred, score = detect_blur(image)

# 输出结果
if is_blurred:
    print(f"Image is blurry. Laplacian variance: {score}")
else:
    print(f"Image is clear. Laplacian variance: {score}")

import cv2
import numpy as np

# 使用傅里叶变换检测模糊
def detect_blur_fft(image, threshold=0.2):
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 计算傅里叶变换
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    
    # 计算频谱
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    
    # 计算高频分量的比例
    rows, cols = gray.shape
    crow, ccol = rows // 2 , cols // 2
    high_freq = np.sum(np.abs(magnitude_spectrum[crow-30:crow+30, ccol-30:ccol+30]))
    total = np.sum(np.abs(magnitude_spectrum))
    
    # 高频比例
    ratio = high_freq / total
    
    # 判断是否模糊
    if ratio < threshold:
        return True, ratio  # 图像模糊
    else:
        return False, ratio  # 图像清晰

# 读取图像
image = cv2.imread('blurred.png')

# 判断图像是否模糊
is_blurred, ratio = detect_blur_fft(image)

# 输出结果
if is_blurred:
    print(f"Image is blurry. High-frequency ratio: {ratio}")
else:
    print(f"Image is clear. High-frequency ratio: {ratio}")

image = cv2.imread('test.png')

# 判断图像是否模糊
is_blurred, ratio = detect_blur_fft(image)

# 输出结果
if is_blurred:
    print(f"Image is blurry. High-frequency ratio: {ratio}")
else:
    print(f"Image is clear. High-frequency ratio: {ratio}")


import cv2
import numpy as np

# Sobel算子模糊检测
def detect_blur_sobel(image, threshold=100):
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 计算Sobel算子的梯度（X方向和Y方向）
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度的绝对值
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 计算梯度的平均值
    mean_grad = np.mean(grad_magnitude)
    
    # 根据平均梯度值判断是否模糊
    if mean_grad < threshold:
        return True, mean_grad  # 图像模糊
    else:
        return False, mean_grad  # 图像清晰

# 读取图像
image = cv2.imread('blurred.jpg')

# 判断图像是否模糊
is_blurred, mean_grad = detect_blur_sobel(image)

# 输出结果
if is_blurred:
    print(f"Image is blurry. Mean gradient magnitude: {mean_grad}")
else:
    print(f"Image is clear. Mean gradient magnitude: {mean_grad}")



image = cv2.imread('test.jpg')

# 判断图像是否模糊
is_blurred, mean_grad = detect_blur_sobel(image)

# 输出结果
if is_blurred:
    print(f"Image is blurry. Mean gradient magnitude: {mean_grad}")
else:
    print(f"Image is clear. Mean gradient magnitude: {mean_grad}")
