# Copyright (c) 2023,Semin Kim, AI R&D Center, lululab

import numpy as np
import os
import cv2
from tqdm import tqdm

if __name__ == '__main__':

    path_src = '/mnt/hdd1/db/wrinkle/AIIM_2023_Rev/crop_2/GT'
    list_src = os.listdir(path_src)

    # Set lengths, If DRRIVE? Set lengths to 584
    len_h = 640
    len_w = 640

    for step_i, str_src in enumerate(tqdm(list_src)):

        fns_src = '%s/%s' % (path_src, str_src)
        img_src = cv2.imread(fns_src, cv2.IMREAD_GRAYSCALE)

        val_rsz = 2
        for i in range(2, 5):
            fns_dst = '%s/%s_%d.png' % (path_src, str_src[:-4], i)

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

