# Copyright (c) 2023,Semin Kim, AI R&D Center, lululab
import albumentations
import torch
import torch.utils.data
import numpy as np
import cv2
import os

class Dataset_album_DRIVE_AIIM_wds_strong_aug(torch.utils.data.Dataset):
    def __init__(self, path_src, path_lbl, b_aug=False, max_pixel=255):
        self.img_dir = path_src
        self.label_dir = path_lbl

        self.b_aug = b_aug
        self.max_pixel = max_pixel

        self.meanRGB = [0.5, 0.5, 0.5]
        self.stdRGB = [0.5, 0.5, 0.5]

        self.examples = []
        file_names = os.listdir(self.img_dir)

        for file_name in file_names:
            example = {}
            example["path_img"] = self.img_dir
            example["path_lbl"] = self.label_dir
            example["str_img"] = file_name
            self.examples.append(example)

        self.num_examples = len(self.examples)

        if b_aug:
            self.transforms = albumentations.Compose([
                albumentations.HorizontalFlip(),
                albumentations.ShiftScaleRotate(),
                albumentations.RandomContrast(limit=0.1),
                albumentations.Normalize(mean=self.meanRGB, std=self.stdRGB, max_pixel_value=self.max_pixel),
            ])
        else:
            self.transforms = albumentations.Compose([
                albumentations.Normalize(mean=self.meanRGB, std=self.stdRGB, max_pixel_value=self.max_pixel),
            ])

    def set_mean_std(self, meanRGB, stdRGB):
        for i in range(3):
            self.meanRGB[i] = meanRGB[i]
            self.stdRGB[i] = stdRGB[i]

    def __getitem__(self, index):

        example = self.examples[index]
        path_img = example['path_img']
        path_lbl = example['path_lbl']
        str_img = example['str_img']

        fns_img = '%s/%s' % (path_img, str_img)
        img_src = cv2.imread(fns_img, cv2.IMREAD_UNCHANGED)
        img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)

        fns_lbl = '%s/%s' % (path_lbl, str_img)
        img_lbl = cv2.imread(fns_lbl, cv2.IMREAD_GRAYSCALE)
        img_lbl = img_lbl / 255

        fns_wd2 = '%s/%s_2.png' % (path_lbl, str_img[:-4])
        fns_wd3 = '%s/%s_3.png' % (path_lbl, str_img[:-4])
        fns_wd4 = '%s/%s_4.png' % (path_lbl, str_img[:-4])

        img_wd2 = cv2.imread(fns_wd2, cv2.IMREAD_GRAYSCALE)
        img_wd3 = cv2.imread(fns_wd3, cv2.IMREAD_GRAYSCALE)
        img_wd4 = cv2.imread(fns_wd4, cv2.IMREAD_GRAYSCALE)

        transforms = self.transforms(image=img_src, masks=[img_lbl, img_wd2, img_wd3, img_wd4])

        src = transforms['image']
        src = np.transpose(src, [2, 0, 1])
        src = src.astype(np.float32)
        msk = transforms['masks'][0]
        wd2 = transforms['masks'][1]
        wd3 = transforms['masks'][2]
        wd4 = transforms['masks'][3]

        # convert numpy -> torch
        src = torch.from_numpy(src)
        msk = torch.from_numpy(msk)
        wd2 = torch.from_numpy(wd2) / 255.
        wd3 = torch.from_numpy(wd3) / 255.
        wd4 = torch.from_numpy(wd4) / 255.

        return src, msk, wd2, wd3, wd4

    def __len__(self):
        return self.num_examples

