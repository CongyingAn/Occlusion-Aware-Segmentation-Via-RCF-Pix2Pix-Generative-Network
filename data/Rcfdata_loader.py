from torch.utils import data
import os
from os.path import join, abspath, splitext, split, isdir, isfile
from PIL import Image
import numpy as np
import cv2



def prepare_image_PIL(im):
    # print(im.shape)
    im = im[:,:,::-1] - np.zeros_like(im) # rgb to bgr
    im -= np.array((104.00698793,116.66876762,122.67891434))
    im = np.transpose(im, (2, 0, 1)) # (H x W x C) to (C x H x W)
    return im

# def prepare_image_PIL(im):
#     if len(im.shape) == 3 and im.shape[2] == 3:  # 检查是否为三通道图像
#         im = im[:, :, ::-1] - np.zeros_like(im)  # RGB to BGR
#         im -= np.array((104.00698793, 116.66876762, 122.67891434))
#         im = np.transpose(im, (2, 0, 1))  # (H x W x C) to (C x H x W)
#     else:  # 灰度图像或单通道图像
#         im = im - np.zeros_like(im)  # 归一化
#         if len(im.shape) == 2:  # 单通道图像
#             im = np.expand_dims(im, axis=0)  # 增加一个维度
#         elif len(im.shape) == 3:  # 灰度图像
#             im = np.expand_dims(im, axis=0)  # 增加一个维度
#             im = np.expand_dims(im, axis=0)  # 再增加一个维度
#     return im

def prepare_image_cv2(im):
    im -= np.array((104.00698793,116.66876762,122.67891434))
    im = np.transpose(im, (2, 0, 1)) # (H x W x C) to (C x H x W)
    return im


class BSDS_RCFLoader(data.Dataset):
    """
    Dataloader BSDS500
    """
    def __init__(self, root='data/HED-BSDS_PASCAL', split='train', transform=False):
        self.root = root
        self.split = split
        self.transform = transform
        if self.split == 'train':
            self.filelist = join(self.root, 'bsds_pascal_train_pair.lst')
        elif self.split == 'test':
            self.filelist = join(self.root, 'test.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()
        # with open(self.filelist, 'r', encoding='latin-1') as file:
        #     self.filelist = file.readlines()

    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):

        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()
            lb = np.array(cv2.imread(join(self.root, lb_file)), dtype=np.float32)
            # cv2.imwrite("test_img\\3.jpg", lb)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb>0, lb<128)] = 2
            lb[lb >= 128] = 1




            img = np.array(cv2.imread(join(self.root, img_file)), dtype=np.float32)
            # cv2.imwrite("test_img\\1.jpg",img)
            img = prepare_image_cv2(img)
            img2 = img.copy()
            return img, lb
        else:

            img_file = self.filelist[index].rstrip()


            # print(img_file)
            # print(self.root)
            img = np.array(Image.open(join(self.root, img_file)), dtype=np.float32)
            # img = np.array(Image.open(img_file), dtype=np.float32)
            # cv2.imwrite("test_img\\2.jpg",img)
            # print(img_file)
            img = prepare_image_PIL(img)
            img2=img.copy()
            return img

