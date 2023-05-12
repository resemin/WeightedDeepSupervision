# Copyright (c) 2023,Semin Kim, AI R&D Center, lululab

import albumentations
import torch
import torch.utils.data
import numpy as np
import cv2

class Dataset_Wrinkle_WDS(torch.utils.data.Dataset):
    def __init__(self, list_src, path_src, path_lbl, path_ttr, b_aug=False, max_pixel=255, height=640, width=640):
        self.list_src = list_src
        self.path_src = path_src
        self.path_lbl = path_lbl
        self.path_ttr = path_ttr

        self.b_aug = b_aug
        self.max_pixel = max_pixel

        self.meanRGB = [0.5, 0.5, 0.5]
        self.stdRGB = [0.25, 0.25, 0.25]

        self.examples = []

        for str_src in list_src:
            example = {}
            example["path_img"] = self.path_src
            example["path_lbl"] = self.path_lbl
            example["path_ttr"] = self.path_ttr
            example["str_img"] = str_src

            self.examples.append(example)

        self.num_examples = len(self.examples)

        if b_aug:
            self.transforms = albumentations.Compose([
                albumentations.Resize(height=height, width=width),
                albumentations.HorizontalFlip(),
                albumentations.ColorJitter(),
            ])
        else:
            self.transforms = albumentations.Compose([
                albumentations.Resize(height=height, width=width)
            ])

        self.transforms_norm = albumentations.Compose([
                albumentations.Normalize(mean=self.meanRGB, std=self.stdRGB, max_pixel_value=self.max_pixel),
            ])

        self.transforms_gray = albumentations.Compose([
            albumentations.Normalize(mean=self.meanRGB[0], std=self.stdRGB[0], max_pixel_value=self.max_pixel),
        ])

    def set_mean_std(self, meanRGB, stdRGB):
        for i in range(3):
            self.meanRGB[i] = meanRGB[i]
            self.stdRGB[i] = stdRGB[i]

    def __getitem__(self, index):

        example = self.examples[index]
        path_img = example['path_img']
        path_lbl = example['path_lbl']
        path_ttr = example['path_ttr']
        str_img = example['str_img']

        fns_img = '%s/%s' % (path_img, str_img)
        img_src = cv2.imread(fns_img, cv2.IMREAD_UNCHANGED)
        img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)

        str_img = str_img[:-4]
        fns_lbl = '%s/%s.png' % (path_lbl, str_img)
        img_lbl = cv2.imread(fns_lbl, cv2.IMREAD_GRAYSCALE)

        fns_ttr = '%s/%s.png' % (path_ttr, str_img)
        img_ttr = cv2.imread(fns_ttr, cv2.IMREAD_GRAYSCALE)

        fns_wds_2 = '%s/%s_2.png' % (path_lbl, str_img)
        img_wds_2 = cv2.imread(fns_wds_2, cv2.IMREAD_GRAYSCALE)

        fns_wds_3 = '%s/%s_3.png' % (path_lbl, str_img)
        img_wds_3 = cv2.imread(fns_wds_3, cv2.IMREAD_GRAYSCALE)

        fns_wds_4 = '%s/%s_4.png' % (path_lbl, str_img)
        img_wds_4 = cv2.imread(fns_wds_4, cv2.IMREAD_GRAYSCALE)

        img_lbl[img_lbl > 0] = 1

        transforms = self.transforms(image=img_src, masks=[img_ttr, img_lbl, img_wds_2, img_wds_3, img_wds_4])
        img_src = transforms['image']
        img_ttr = transforms['masks'][0]
        img_lbl = transforms['masks'][1]
        inp_wds_2 = transforms['masks'][2]
        inp_wds_3 = transforms['masks'][3]
        inp_wds_4 = transforms['masks'][4]

        transforms_img = self.transforms_norm(image=img_src, mask=img_lbl)
        transforms_ttr = self.transforms_gray(image=img_ttr, mask=img_lbl)

        inp_src = transforms_img['image']
        inp_lbl = transforms_img['mask']
        inp_ttr = transforms_ttr['image']

        inp_src = np.transpose(inp_src, [2, 0, 1])
        inp_ttr = np.expand_dims(inp_ttr, 0)

        inp_src = inp_src.astype(np.float32)
        inp_ttr = inp_ttr.astype(np.float32)
        inp_wds_2 = inp_wds_2.astype(np.float32) / 255.
        inp_wds_3 = inp_wds_3.astype(np.float32) / 255.
        inp_wds_4 = inp_wds_4.astype(np.float32) / 255.

        # convert numpy -> torch
        inp_src = torch.from_numpy(inp_src)
        inp_lbl = torch.from_numpy(inp_lbl)
        inp_ttr = torch.from_numpy(inp_ttr)
        inp_wds_2 = torch.from_numpy(inp_wds_2)
        inp_wds_3 = torch.from_numpy(inp_wds_3)
        inp_wds_4 = torch.from_numpy(inp_wds_4)

        return inp_src, inp_lbl, inp_ttr, inp_wds_2, inp_wds_3, inp_wds_4

    def __len__(self):
        return self.num_examples
