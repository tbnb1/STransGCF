import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import RandomHorizontalFlip, ColorJitter

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label
# 自定义添加高斯噪声的函数
def add_gaussian_noise(img, mean=0, std=0.1):
    # 将PIL图像转换为numpy数组
    img = np.array(img)
    # 生成高斯噪声
    noise = np.random.normal(mean, std, img.shape)
    # 将噪声添加到图像上
    noisy_img = img + noise * 255
    # 将噪声图像裁剪到[0, 255]范围内
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    # 将numpy数组转换回PIL图像
    return Image.fromarray(noisy_img)

# 定义一个自定义的数据增广类，它随机选择一种变换
class RandomAugmentation(object):
    def __init__(self):
        super().__init__()

    def __call__(self, img):
        # 随机选择一个变换索引
        transform_idx = torch.randint(0, 4, (1,)).item()
        # 应用所选的变换
        if transform_idx == 0:
            return img
        elif transform_idx == 1:
            return ColorJitter(brightness=0.5)(img)
        elif transform_idx == 2:
            return ColorJitter(saturation=0.5)(img)
        elif transform_idx == 3:
            return add_gaussian_noise(img)
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        # elif random.random() > 0.5:
        #    image, label = random_rotate(image, label)
        image = Image.fromarray(image)
        label = Image.fromarray(label)
            # image, label = random_rot_flip(image, label)
        augmentation = RandomAugmentation()
        image = augmentation(image)
        # elif random.random() > 0.5:
        #     image, label = random_rotate(image, label)
        # x, y, _ = image.shape
        # if x != self.output_size[0] or y != self.output_size[1]:
        #     image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)  # why not 3?
        #     label = zoom(label, (self.output_size[  0] / x, self.output_size[1] / y), order=0)
        sample = {'image': image, 'label': label}
        # print(image.shape)
        return sample

class mydata_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir
        self.norm = transforms.Normalize(mean=[203.37, 135.97, 209.36], std=[21.27, 22.97, 11.87])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx].strip('\n')
        img_path = os.path.join(self.data_dir, "img/" + slice_name)
        label_path = os.path.join(self.data_dir, "labelcol/" + slice_name)
        with Image.open(img_path) as img:
            image = np.array(img)
        with Image.open(label_path) as lbl:
            label = np.array(lbl)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        image = sample['image']
        label = sample['label']
        image = torch.from_numpy(np.array(image).astype(np.float32)).permute(2, 0, 1)
        label = torch.from_numpy(np.array(label).astype(np.float32))

        # 测试之前的模型记得把下面这一行注释掉
        image = self.norm(image)

        label[label > 0] = 1
        sample = {'image': image, 'label': label}
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample