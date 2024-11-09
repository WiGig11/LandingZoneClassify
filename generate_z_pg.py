from PIL import Image
import random
import os

"""
generate random captured playground image
"""
image_path = '90.jpg'  # 替换为你的图像文件路径
output_dir = './Landing_dataset/Z'  # 输出图像的文件路径
times = 300
# 打开图像
img = Image.open(image_path)

# 确保图像足够大
if img.width < 20 or img.height < 20:
    raise ValueError("Image is too small to crop a 20x20 area.")

for time in range(times):
    
# 随机选择裁剪区域的起始点
    left = random.randint(0, img.width - 20)
    top = random.randint(0, img.height - 20)
    # 裁剪20x20的区域
    box = (left, top, left + 20, top + 20)
    region = img.crop(box)

    # 将裁剪的区域放大到100x100
    resized_region = region.resize((100, 100), Image.LANCZOS)

    # 保存放大后的图像
    output_path = os.path.join(output_dir,f'{time}.png') 
    resized_region.save(output_path)
