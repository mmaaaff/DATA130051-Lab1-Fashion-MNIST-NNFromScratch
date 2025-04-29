from MyDL.tensor import *
import cupy as np
import random
from PIL import Image, ImageEnhance
from typing import Tuple

class Dataset():
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def Dataloader(dataset, batch_size, shuffle=True):
    if isinstance(dataset, MyTensor):
        raise(TypeError("Dataset must be an instance of MyDL.data.Dataset instead of MyDL.tensor.MyTensor."))
    n = len(dataset)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    j = 0
    for i in range(0, n - batch_size + 1, batch_size):
        j = i + batch_size
        batch_indices = indices[i:j]
        yield dataset[batch_indices]
    if j < n:
        batch_indices = indices[j:]
        yield dataset[batch_indices]


class mnist_dataset(Dataset):
    def __init__(self, images:MyTensor, labels, augment=False, augment_prob=0.5, unfold=False):
        self.images = images
        self.labels = labels
        self.augment = augment
        self.augment_prob = augment_prob
        self.unfold = unfold

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        images = self.images[index]
        labels = self.labels[index]

        # Data Augmentation
        if self.augment:
            images = images.data
            assert images.dtype == np.uint8, "Image dtype must be uint8 for augmentation."
            hflip_prob = 0.5
            rotate_degrees = 45
            brightness_range = (0.8, 1.2)

            n, c, h, w = images.shape
            augmented_images_list = []

            for i in range(n):
                do_augentation = random.random() < self.augment_prob
                if not do_augentation:
                    augmented_images_list.append(images[i])
                    continue

                single_image_chw = np.asnumpy(images[i])  #(c, h, w)

                img_np_hw = single_image_chw[0]  # (h, w)

                pil_mode = 'L'

                img_pil = Image.fromarray(img_np_hw, mode=pil_mode)

                # --- Augmentation ---
                # Flipping
                # if random.random() < hflip_prob:
                #     img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
                # Rotating
                if rotate_degrees > 0:
                    angle = random.uniform(-rotate_degrees, rotate_degrees)
                    # 使用 BICUBIC 插值获得更好的旋转质量
                    fill_value = (0,) * c
                    img_pil = img_pil.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=fill_value)
                # Brightness Adjustment
                if brightness_range != (1.0, 1.0):
                    factor = random.uniform(brightness_range[0], brightness_range[1])
                    enhancer = ImageEnhance.Brightness(img_pil)
                    img_pil = enhancer.enhance(factor)
                # Contrast Adjustment
                if random.random() < 0.8:
                    contrast_factor = random.uniform(0.8, 1.2)
                    enhancer = ImageEnhance.Contrast(img_pil)
                    img_pil = enhancer.enhance(contrast_factor)
                # Color Adjustment
                if random.random() < 0.8:
                    color_factor = random.uniform(0.8, 1.2)
                    enhancer = ImageEnhance.Color(img_pil)
                    img_pil = enhancer.enhance(color_factor)

                augmented_img_np_chw = np.expand_dims(np.array(img_pil), axis=0)  # (1, h, w)
                augmented_images_list.append(augmented_img_np_chw)
            images = np.array(augmented_images_list)
            if self.unfold:
                images = images.reshape(-1, 28 * 28)
            images = MyTensor(images, requires_grad=False)  # transfer to gpu and create tensor
        
        else:
            images = images.data
            if self.unfold:
                images = images.reshape(-1, 28 * 28)
            images = MyTensor(np.array(images), requires_grad=False)

        return images, labels