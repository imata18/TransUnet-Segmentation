import torch
import torch.utils.data as data
from torch.autograd import Variable as V
from PIL import Image

import cv2
import numpy as np
import os
import scipy.misc as misc

# data augmentation

# TODO: might need to scale the img and mask to meet the required size


def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask


# The mode could be train, val and test, so we can read from different folders

def read_own_data(root_path, mode='train'):
    images = []
    masks = []

    image_root = os.path.join(root_path, mode + '/images')
    mask_root = os.path.join(root_path, mode + "/labels")

    for img_name in os.listdir(image_root):
        image_path = os.path.join(image_root, img_name)
        label_path = os.path.join(mask_root, img_name)

        images.append(image_path)
        masks.append(label_path)

    return images, masks


def mask_to_onehot(mask, palette):
    semantic_map = []
    for color in palette:
        equality = np.equal(mask, color)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map


def own_data_loader(img_path, mask_path):
    palette = [[0], [252], [253], [254], [255]]
    img = cv2.imread(img_path, 0)
    mask = cv2.imread(mask_path, 0)

    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)

# since the img and maks are both grayscale, dimension expansion may be needed,
# depends on the network input
#     print("before expand dims", img.shape)

    img = np.expand_dims(img, axis=2)
    mask = np.expand_dims(mask, axis=2)

    # convert the masks from(H,W,1) to (H,W,5)
    mask = mask_to_onehot(mask, palette)
    # print('img.shape', img.shape)
    # print('mask.shape', mask.shape)

    img = np.array(img, np.float32).transpose([2, 0, 1])  # covert to CHW
    mask = np.array(mask, np.float32).transpose([2, 0, 1])
    return img, mask


def own_data_test_loader(img_path, mask_path):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)

    return img, mask


class ImageFolder(data.Dataset):

    def __init__(self, root_path, mode='train'):
        self.root = root_path
        self.mode = mode
        self.images, self.labels = read_own_data(self.root, self.mode)

    def __getitem__(self, index):
        if self.mode == 'test':
            img, mask = own_data_test_loader(self.images[index], self.labels[index])
        else:
            img, mask = own_data_loader(self.images[index], self.labels[index])
            img = torch.Tensor(img)
            mask = torch.Tensor(mask)
        return img, mask

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)

