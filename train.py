import os
import tqdm
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import default_collate
import pdb
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('test.png')
    plt.close()



# 自定义MNIST数据集
class CustomMNIST(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = img.numpy().astype('uint8')
        img = torchvision.transforms.functional.to_pil_image(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

# 自定义collate函数
def custom_collate_stage1(batch):
    new_batch = []
    for items in batch:
        image, label = items
        #pdb.set_trace()
        label_tensor = torch.tensor(int(label), dtype=torch.long)
        new_batch.append((image, label_tensor))
    return default_collate(new_batch)
    
class Model2(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 50)
        self.fc4 = nn.Linear(50, num_classes)
        self.ac = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.ac(self.conv1(x)))
        x = self.pool(self.ac(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #pdb.set_trace()#
        x = self.ac(self.fc1(x))
        x = self.ac(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        return x
        
# 定义模型结构（简单的CNN模型）
class Model(nn.Module):
    def __init__(self, num_classes=2):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 输入通道数为1
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),  # 根据输入尺寸调整
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 测试函数
def test(model, testloader, device, phase='Test'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm.tqdm(testloader, desc=f'{phase}'):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            #pdb.set_trace()
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the {phase} images: {accuracy:.2f} %')
    return accuracy

# 训练函数
def train(device):
    # 数据增强和预处理
    from PIL import Image
    from torchvision import transforms
    from torch.utils.data import Dataset, DataLoader
    import os
    
    # 定义 transform，包括旋转、模糊、灰度化、调整大小等随机增强
    rotation_transforms = transforms.RandomChoice([
        transforms.RandomRotation([0, 0]),
        transforms.RandomRotation([90, 90]),
        transforms.RandomRotation([180, 180]),
        transforms.RandomRotation([270, 270])
    ])
    
    blur_transform = transforms.RandomChoice([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),  # 轻微模糊
        transforms.GaussianBlur(kernel_size=5, sigma=(1.0, 2.0)),  # 中度模糊
    ])
    
    transform_custom = transforms.Compose([
        rotation_transforms,
        blur_transform,
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 创建自定义 Dataset，用于动态生成增强版本
    class AugmentedDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.image_paths = []
            self.labels = []
    
            # 遍历每个子文件夹，收集图像路径和标签
            for label, folder_name in enumerate(sorted(os.listdir(root_dir))):
                folder_path = os.path.join(root_dir, folder_name)
                if os.path.isdir(folder_path):
                    for filename in os.listdir(folder_path):
                        image_path = os.path.join(folder_path, filename)
                        self.image_paths.append(image_path)
                        self.labels.append(label)
    
        def __len__(self):
            # 设置一个很大的值，模拟无限增强数据集
            return 100000  # 可以根据需要设置为任意大
    
        def __getitem__(self, idx):
            # 随机选择一个图像路径和标签
            real_idx = idx % len(self.image_paths)  # 防止越界
            image_path = self.image_paths[real_idx]
            label = self.labels[real_idx]
            
            # 加载图像并应用 transform
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
    
            return image, label
    class AugmentedDataset2(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.image_paths = []
            self.labels = []
    
            # 遍历每个子文件夹，收集图像路径和标签
            # 遍历每个子文件夹，收集图像路径和标签
            for label, folder_name in enumerate(sorted(os.listdir(root_dir))):
                folder_path = os.path.join(root_dir, folder_name)
                if os.path.isdir(folder_path):
                    for filename in os.listdir(folder_path):
                        image_path = os.path.join(folder_path, filename)
                        self.image_paths.append(image_path)
                        self.labels.append(label)

    
        def __len__(self):
            # 设置一个很大的值，模拟无限增强数据集
            return 1000  # 可以根据需要设置为任意大
    
        def __getitem__(self, idx):
            # 随机选择一个图像路径和标签
            real_idx = idx % len(self.image_paths)  # 防止越界
            image_path = self.image_paths[real_idx]
            label = self.labels[real_idx]
            #pdb.set_trace()
            
            # 加载图像并应用 transform
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
    
            return image, label
    # 使用自定义 Dataset 和 DataLoader
    dataset = AugmentedDataset(root_dir="./Landing_dataset", transform=transform_custom)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 测试 DataLoader 输出
    for images, labels in dataloader:
        print(f"Batch of images shape: {images.shape}")
        print(f"Batch of labels: {labels}")
        break  # 仅打印第一个批次以测试效果

    # 加载字母数据（标签为0）
    letters_dataset = ImageFolder(root="./subtle_augmentation_images", transform=transform_custom)
    #letters_labels = [0] * len(letters_dataset)  # 标记为0（字母）

    # 加载数字数据（标签为1）
    #mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    #target_digits = [1, 2, 3]
    #idx_train = (mnist_train.targets == target_digits[0]) | (mnist_train.targets == target_digits[1]) | \
    #            (mnist_train.targets == target_digits[2])
    #mnist_data_train = mnist_train.data[idx_train]
    #mnist_targets_train = 26 * torch.ones(len(mnist_data_train), dtype=torch.long)  # 标记为26（数字dustin 维度）
    #mnist_targets_train = (mnist_train.targets[idx_train]) + 6
    #custom_mnist_dataset_train = CustomMNIST(mnist_data_train, mnist_targets_train, transform=transform)

    # 合并数据集
    #train_dataset_stage1 = ConcatDataset([letters_dataset, custom_mnist_dataset_train])
    train_dataset_stage1 = letters_dataset
    train_loader_stage1 = DataLoader(dataset=train_dataset_stage1, batch_size=8, shuffle=True, collate_fn=custom_collate_stage1)
    #train_loader_stage1 = DataLoader(dataset=train_dataset_stage1, batch_size=128, shuffle=True)
    #pdb.set_trace()

    # 定义第一阶段模型和损失函数
    letter_detector = Model2(num_classes=11).to(device)
    criterion_stage1 = nn.CrossEntropyLoss()
    optimizer_stage1 = optim.Adam(letter_detector.parameters(), lr=0.001)

    # 训练第一阶段模型
    print("Training Letter Detector (Stage 1)")
    for epoch in range(20):  # 可根据需要调整训练轮数
        letter_detector.train()
        running_loss = 0.0
        for data in tqdm.tqdm(dataloader, desc=f'Epoch {epoch+1}/20'):
            # get some random training images
#dataiter = iter(trainloader)
#images, labels = next(dataiter)

# show images
            inputs, labels = data[0].to(device), data[1].to(device)
#pdb.set_trace()
            #imshow(torchvision.utils.make_grid(inputs.cpu()))
# print labels
            #print(labels)
            
            #pdb.set_trace()
            optimizer_stage1.zero_grad()
            outputs = letter_detector(inputs)
           # pdb.set_trace()
            loss = criterion_stage1(outputs, labels)
            loss.backward()
            optimizer_stage1.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader_stage1):.4f}')
    # -----------------------------
    # 测试阶段
    # -----------------------------

    # 测试数据准备
    print("Preparing Test Data")
    # 加载测试字母数据集（标签为0）
    dataset = AugmentedDataset2(root_dir="./Landing_dataset", transform=transform_custom)
    dataloadertest = DataLoader(dataset, batch_size=64, shuffle=True)
    letters_dataset_test = ImageFolder(root="./subtle_augmentation_images", transform=transform_custom)
    # 如果没有单独的测试集，可以拆分部分训练集作为测试
    #letters_labels_test = [0] * len(letters_dataset_test)  # 标记为0（字母）

    # 加载测试数字数据集（标签为1）
    #mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    #idx_test = (mnist_test.targets == 1) | (mnist_test.targets == 2) | (mnist_test.targets == 3)
    #mnist_data_test = mnist_test.data[idx_test]
    #mnist_targets_test = 26 * torch.ones(len(mnist_data_test), dtype=torch.long)  # 标签为1（数字）
    #mnist_targets_test = (mnist_test.targets[idx_test]) + 6
    #custom_mnist_dataset_test = CustomMNIST(mnist_data_test, mnist_targets_test, transform=transform)

    # 合并测试数据集
    #test_dataset_stage1 = ConcatDataset([letters_dataset_test, custom_mnist_dataset_test])
    test_dataset_stage1 = letters_dataset_test
    #test_loader_stage1 = DataLoader(dataset=test_dataset_stage1, batch_size=128, shuffle=False)
    test_loader_stage1 = DataLoader(dataset=test_dataset_stage1, batch_size=128, shuffle=False, collate_fn=custom_collate_stage1)

    # 测试第一阶段模型
    print("Testing Letter Detector (Stage 1)")
    test(letter_detector, dataloadertest, device, phase='Stage 1 Test')
    
    # 保存模型
    print('Finished Training and Testing')
    PATH = 'models'
    os.makedirs(PATH, exist_ok=True)
    torch.save(letter_detector.state_dict(), os.path.join(PATH, 'letter_detector.pth'))
    
# 主函数
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train(device)
