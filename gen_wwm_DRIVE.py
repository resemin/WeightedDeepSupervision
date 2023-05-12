# Copyright (c) 2023,Semin Kim, AI R&D Center, lululab

import numpy as np
import os
import cv2
from tqdm import tqdm
import shutil

if __name__ == '__main__':

    path_src = 'Retinal/DRIVE/training/manual_rename_png'
    path_dst = 'Retinal/DRIVE/training/manual_rename_png_584_584'

    if os.path.isdir(path_dst) == False:
        os.makedirs(path_dst)

    list_src = os.listdir(path_src)

    len_h = 584
    len_w = 584

    for step_i, str_src in enumerate(tqdm(list_src)):

        fns_src = '%s/%s' % (path_src, str_src)
        fns_dst = '%s/%s' % (path_dst, str_src)
        shutil.copyfile(fns_src, fns_dst)
        img_src = cv2.imread(fns_src, cv2.IMREAD_GRAYSCALE)
        img_src = np.concatenate([np.zeros((584, 9), dtype=np.uint8), img_src, np.zeros((584, 10), dtype=np.uint8)], axis=1)
        val_rsz = 2
        for i in range(2, 5):
            fns_dst = '%s/%s_%d.png' % (path_dst, str_src[:-4], i)

            new_h = int(len_h / val_rsz)
            new_w = int(len_w / val_rsz)

            img_apl = np.zeros((new_h, new_w), dtype=np.uint8)

            for y in range(new_h):
                y_s = y * val_rsz
                y_e = y_s + val_rsz
                for x in range(new_w):
                    x_s = x * val_rsz
                    x_e = x_s + val_rsz

                    img_apl[y, x] = np.mean(np.mean(img_src[y_s:y_e, x_s:x_e]))

            img_ups = cv2.resize(img_apl, (len_w, len_h), interpolation=cv2.INTER_NEAREST)
            pix_nonzero = img_ups[img_ups > 0]
            mean_nonzero = np.mean(pix_nonzero)
            img_ups[img_ups == 0] = mean_nonzero
            img_ups[img_src == 255] = 255
            cv2.imwrite(fns_dst, img_ups)

            val_rsz = val_rsz * 2

