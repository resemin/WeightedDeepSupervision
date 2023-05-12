# Copyright (c) 2023,Semin Kim, AI R&D Center, lululab

import cv2
import os
import glob
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':

    # Edit your path
    path_src = '/mnt/hdd1/db/wrinkle/AIIM_2023_Rev/crop_2/src'
    path_dst = '/mnt/hdd1/db/wrinkle/AIIM_2023_Rev/crop_2/texture'
    if os.path.isdir(path_dst) == False:
        os.makedirs(path_dst)

    # Get list of images
    list_png = glob.glob('%s/*.png' % path_src)
    list_png.sort()

    # Set Gaussian Kernel
    kernel1d = cv2.getGaussianKernel(21, 5)
    kernel2d = np.outer(kernel1d, kernel1d.transpose())

    # Get textures and save
    for fns_src in tqdm(list_png):
        fns_dst = fns_src.replace(path_src, path_dst)

        img_src = cv2.imread(fns_src, cv2.IMREAD_GRAYSCALE)
        img_src = np.array(img_src, dtype=float)
        img_low = cv2.filter2D(img_src, -1, kernel2d)
        img_low = np.array(img_low, dtype=float)

        img_div = (img_src * 255.) / (img_low + 1.)
        img_div[img_div > 255.] = 255.
        img_div = np.array(img_div, dtype=np.uint8)
        img_div = 1 - img_div

        cv2.imwrite(fns_dst, img_div)

    print('finished')
