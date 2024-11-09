from PIL import Image, ImageDraw, ImageFont, ImageFilter,ImageOps
import numpy as np
import os
import random
import pdb

# 创建输出文件夹
output_dir = "Landing_dataset"
os.makedirs(output_dir, exist_ok=True)

# 图像尺寸和字体设置
image_size = (100, 100)  # 200x200像素的正方形
hex_color = "#363230"

# 去掉 '#' 符号并将十六进制分量转换为 RGB
rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
background_color = rgb_color
text_color = (255, 255, 255)  # 白色文字
#border_color = (255, 255, 255)  # 白色边框
letters = "TYWXFOP"  # 要生成的字母
nums = "123"
font_size = 99  # 字体大小
times = 300

# 尝试加载系统字体，若失败则使用默认字体
try:
    font = ImageFont.truetype("euclidb.ttf", font_size)#CENTURY.TTF cambriai.ttf
except IOError:
    font = ImageFont.load_default()
#fonts = ["euclidb.ttf", "CENTURY.TTF", "cambriai.ttf"]
#fonts = ["CENTURY.ttf"]
# 生成每个字母的图像并应用增强
for letter in letters:
    for time in range(0,times):
        # 创建图像
        #font = random.choice(fonts)
        font = ImageFont.truetype("BodoniFLF-Bold.ttf", font_size)
        img = Image.new("RGB", image_size, background_color)
        draw = ImageDraw.Draw(img)

                # 计算文字的尺寸和位置，以便居中绘制
        bbox = draw.textbbox((0, 0), letter, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # 计算绘制位置，将文字居中
        text_x = (image_size[0] - text_width) / 2 - bbox[0]
        text_y = (image_size[1] - text_height) / 2 - bbox[1]

        # 绘制文字
        draw.text((text_x, text_y), letter, fill=text_color, font=font)


        # 绘制文字
        draw.text((text_x, text_y), letter, fill=text_color, font=font)

        # 在0, 90, 180, 270度之间随机选择一个旋转角度
        rotation_angle = random.choice([0, 90, 180, 270])
        img = img.rotate(rotation_angle, resample=Image.BICUBIC, expand=False)

        # 轻微模糊
        img = img.filter(ImageFilter.GaussianBlur(radius=1.5))

        # 添加椒盐噪声
        img_array = np.array(img)
        noise_prob = 0.02
        salt_vs_pepper = 0.5
        num_salt = int(noise_prob * img_array.size * salt_vs_pepper)
        num_pepper = int(noise_prob * img_array.size * (1 - salt_vs_pepper))

        coords = [np.random.randint(0, i - 1, num_salt) for i in img_array.shape[:2]]
        img_array[coords[0], coords[1]] = [255, 255, 255]
        coords = [np.random.randint(0, i - 1, num_pepper) for i in img_array.shape[:2]]
        img_array[coords[0], coords[1]] = [0, 0, 0]
        img = Image.fromarray(img_array)

        # 随机添加震动模糊
        # 创建一个简单的运动模糊核
        kernel_size = 5
        angle = random.choice([0, 90])  # 随机在水平或垂直方向上应用
        if angle == 0:  # 水平方向
            kernel = [1/kernel_size] * kernel_size + [0] * (kernel_size * (kernel_size - 1))
            kernel = np.reshape(kernel, (kernel_size, kernel_size)).tolist()
        else:  # 垂直方向
            kernel = [0] * (kernel_size * (kernel_size - 1)) + [1/kernel_size] * kernel_size
            kernel = np.reshape(kernel, (kernel_size, kernel_size)).tolist()

        # 应用运动模糊内核
        kernel = ImageFilter.Kernel((kernel_size, kernel_size), sum(kernel, []), scale=None)
        img = img.filter(kernel)

        #img_with_border = ImageOps.expand(img, border=5, fill=border_color)

        # 保存增强后的图像
        path = os.path.join(output_dir,letter)
        #pdb.set_trace()
        if not os.path.exists(path):
            os.makedirs(path)
            
        img.save(os.path.join(path, f"{time}.png"))
for num in nums:
    for time in range(0,times):
        # 创建图像
        #font = random.choice(fonts)
        font = ImageFont.truetype("euclidb.ttf", font_size)
        img = Image.new("RGB", image_size, background_color)
        draw = ImageDraw.Draw(img)

                # 计算文字的尺寸和位置，以便居中绘制
        bbox = draw.textbbox((0, 0), num, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # 计算绘制位置，将文字居中
        text_x = (image_size[0] - text_width) / 2 - bbox[0]
        text_y = (image_size[1] - text_height) / 2 - bbox[1]

        # 绘制文字
        draw.text((text_x, text_y), num, fill=text_color, font=font)


        # 绘制文字
        draw.text((text_x, text_y), num, fill=text_color, font=font)

        # 在0, 90, 180, 270度之间随机选择一个旋转角度
        rotation_angle = random.choice([0, 90, 180, 270])
        img = img.rotate(rotation_angle, resample=Image.BICUBIC, expand=False)

        # 轻微模糊
        img = img.filter(ImageFilter.GaussianBlur(radius=1.5))

        # 添加椒盐噪声
        img_array = np.array(img)
        noise_prob = 0.02
        salt_vs_pepper = 0.5
        num_salt = int(noise_prob * img_array.size * salt_vs_pepper)
        num_pepper = int(noise_prob * img_array.size * (1 - salt_vs_pepper))

        coords = [np.random.randint(0, i - 1, num_salt) for i in img_array.shape[:2]]
        img_array[coords[0], coords[1]] = [255, 255, 255]
        coords = [np.random.randint(0, i - 1, num_pepper) for i in img_array.shape[:2]]
        img_array[coords[0], coords[1]] = [0, 0, 0]
        img = Image.fromarray(img_array)

        # 随机添加震动模糊
        # 创建一个简单的运动模糊核
        kernel_size = 5
        angle = random.choice([0, 90])  # 随机在水平或垂直方向上应用
        if angle == 0:  # 水平方向
            kernel = [1/kernel_size] * kernel_size + [0] * (kernel_size * (kernel_size - 1))
            kernel = np.reshape(kernel, (kernel_size, kernel_size)).tolist()
        else:  # 垂直方向
            kernel = [0] * (kernel_size * (kernel_size - 1)) + [1/kernel_size] * kernel_size
            kernel = np.reshape(kernel, (kernel_size, kernel_size)).tolist()

        # 应用运动模糊内核
        kernel = ImageFilter.Kernel((kernel_size, kernel_size), sum(kernel, []), scale=None)
        img = img.filter(kernel)

        #img_with_border = ImageOps.expand(img, border=5, fill=border_color)

        # 保存增强后的图像
        path = os.path.join(output_dir,num)
        #pdb.set_trace()
        if not os.path.exists(path):
            os.makedirs(path)
        img.save(os.path.join(path, f"{time}.png"))
print("极轻微增强后的图片已生成并保存在文件夹 'Landing_dataset' 中。")
