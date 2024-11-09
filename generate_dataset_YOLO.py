import torch
from torch.utils.data import ConcatDataset, Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as F
from torchvision.datasets import ImageFolder

import random
from PIL import Image
import matplotlib.pyplot as plt
"""
This is a py to generate dataset for YOLO digits and letters recogonition
这是一个用来生成YOLO的数据集的py文件
"""
class CustomMNIST(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        target = self.targets[index]
        image = F.to_pil_image(image)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(target, dtype=torch.long)

    @property
    def num_classes(self):
        return len(set(self.targets.numpy()))

    @property
    def classes(self):
        return set(self.targets.numpy())

def get_random_transform(dataset, image_size=(224, 224)):
    # 创建一个随机颜色的空白背景图像
    background_color = tuple(random.randint(0, 255) for _ in range(3))
    background = Image.new("RGB", image_size, background_color)
    annotations = []  # 用于存储标注信息
    bboxes = []  # 用于存储已放置数字的边界框

    # 随机生成样本数量，范围为 1 到 3
    sample_number = random.randint(1, 3)

    # 加载数据集
    mnist_data = dataset
    for _ in range(sample_number):
        # 从数据集中随机选择一个样本
        sample_index = random.randint(0, len(mnist_data) - 1)
        target, label = mnist_data[sample_index]
        import pdb
        #pdb.set_trace()
        label = int(str(label)[-2])

        # 将图像转换为 RGBA 模式（以保留背景透明性）
        target_image = target.convert("RGBA")

        # 随机生成缩放比例和旋转角度
        scale_factor = random.uniform(1.6, 2)  # 缩放比例，可根据需要调整范围
        rotation_angle = random.uniform(-30, 30)  # 旋转角度

        # 应用缩放
        new_size = (int(target_image.width * scale_factor), int(target_image.height * scale_factor))
        target_image = target_image.resize(new_size, Image.LANCZOS)  # 使用 Image.LANCZOS

        # 应用旋转
        target_image = target_image.rotate(rotation_angle, expand=True)

        # 更新图像尺寸
        img_width, img_height = target_image.size

        # 确保图像尺寸不超过背景尺寸
        if img_width >= background.width or img_height >= background.height:
            # 如果图像尺寸超过背景尺寸，则跳过此样本
            continue

        # 尝试找到一个不重叠的位置
        max_attempts = 50  # 最大尝试次数
        for attempt in range(max_attempts):
            x = random.randint(0, background.width - img_width)
            y = random.randint(0, background.height - img_height)
            new_bbox = (x, y, x + img_width, y + img_height)

            # 检查是否与已有的边界框重叠
            overlap = False
            for bbox in bboxes:
                # 如果边界框有交集，则认为重叠
                if (new_bbox[0] < bbox[2] and new_bbox[2] > bbox[0] and
                    new_bbox[1] < bbox[3] and new_bbox[3] > bbox[1]):
                    overlap = True
                    break

            if not overlap:
                # 不重叠，放置数字并记录边界框
                background.paste(target_image, (x, y), target_image)
                bboxes.append(new_bbox)

                # 使用实际图像宽高生成注释信息（YOLO 格式: 类别 x_center y_center 宽度 高度）
                annotations.append(
                    (
                        label,
                        (x + img_width / 2) / background.width,
                        (y + img_height / 2) / background.height,
                        img_width / background.width,
                        img_height / background.height,
                    )
                )
                break  # 成功放置，退出尝试循环
        else:
            # 如果在最大尝试次数内未找到不重叠的位置，跳过此数字
            print(f"无法在不重叠的情况下放置数字 {label}，已尝试 {max_attempts} 次。")
            continue

    return background, annotations

def main():
    mnist = datasets.MNIST(root="./data", train=True, download=True)
    target_digits = [1, 2, 3]
    idx = (
        (mnist.targets == target_digits[0])
        | (mnist.targets == target_digits[1])
        | (mnist.targets == target_digits[2])
    )
    mnist_data = mnist.data[idx]
    mnist_targets = mnist.targets[idx]
    custom_mnist_dataset = CustomMNIST(mnist_data, mnist_targets, transform=None)

    # 如果您需要与其他数据集合并
    #TODO:
    train_data2 = ImageFolder(root="./Your_dir", transform=None)
    combined_dataset = ConcatDataset([custom_mnist_dataset, train_data2])

    # 可视化图像及其标注框
    background_image, annotations = get_random_transform(custom_mnist_dataset)
    fig, ax = plt.subplots()
    ax.imshow(background_image)

    # 绘制标注框
    for ann in annotations:
        label, x_center, y_center, width, height = ann
        box_x = (x_center - width / 2) * background_image.width
        box_y = (y_center - height / 2) * background_image.height
        box_width = width * background_image.width
        box_height = height * background_image.height
        rect = plt.Rectangle((box_x, box_y), box_width, box_height, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(box_x, box_y - 5, str(label), color='red', fontsize=12, ha='center')

    plt.axis('off')
    plt.show()

    from concurrent.futures import ThreadPoolExecutor
    import time

#TODO: Change to your dirs
    def save_data1(index, dataset):
        background_image, annotations = get_random_transform(dataset)
        background_image.save(f"dataset/images/test/{index}.png")
        with open(f"dataset/labels/test/{index}.txt", "w") as file:
            for item in annotations:
                file.write(str(item) + "\n")

    def save_data2(index, dataset):
        background_image, annotations = get_random_transform(dataset)
        background_image.save(f"dataset/images/train/{index}.png")
        with open(f"dataset/labels/train/{index}.txt", "w") as file:
            for item in annotations:
                file.write(str(item) + "\n")

    t1 = time.time()
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(save_data1, i, combined_dataset) for i in range(1000)]
    t2 = time.time()
    print(f"Total time: {t2 - t1} seconds")

    t1 = time.time()
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(save_data2, i, combined_dataset) for i in range(8000)]
    t2 = time.time()
    print(f"Total time: {t2 - t1} seconds")

if __name__ == "__main__":
    main()
