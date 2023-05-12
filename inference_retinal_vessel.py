# Copyright (c) 2023,Semin Kim, AI R&D Center, lululab

import torch
from torch.autograd import Variable
from core.models import AG_Net, AG_Net_ASPP
from dataset.datasets_DRIVE import Dataset_album_DRIVE_AIIM
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import random
import sklearn


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed = 3407
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device :', device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())

    num_epochs = 200
    train_batch_size = 4
    val_batch_size = 2
    learning_rate = 0.01
    num_class = 2

    path_va_src = '../../DB/DRIVE/test/images_rename_png_584_584'
    path_va_lbl = '../../DB/DRIVE/test/1st_manual_rename_png_584_584'

    max_pixel = 255
    val_dataset = Dataset_album_DRIVE_AIIM(path_src=path_va_src, path_lbl=path_va_lbl, b_aug=False, max_pixel=max_pixel)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4)

    unet = AG_Net(n_classes=num_class, bn=True, BatchNorm=False).to(device)
    fns_mdl = '/home/semin/outside/gpu1/Users/ksm/Wrinkle/Code/agnet/save_model/Rev_new_E200/unet_epoch_181_cIOU_0.7036.pth'
    unet.load_state_dict(torch.load(fns_mdl), strict=True)

    softmax_2d = torch.nn.Softmax2d()

    unet.eval()
    EPS = 1e-12

    y_true = []
    y_score = []

    for step, (imgs, label_imgs) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            imgs = Variable(imgs).to(device)
            label_imgs = Variable(label_imgs.type(torch.LongTensor)).to(device)

            [side_5, side_6, side_7, side_8] = unet(imgs)

        out = torch.log(softmax_2d(side_8) + EPS)

        score = torch.softmax(out, dim=1).cpu().numpy()
        score = score[:, 1, :, :]

        label_imgs_np = label_imgs.cpu().numpy()

        y_score.append(score.reshape(-1))
        y_true.append(label_imgs_np.reshape(-1))

        y_score_np = np.asarray(y_score).reshape(-1)
        y_pred_np = y_score_np > 0.5
        y_true_np = np.asarray(y_true).reshape(-1)

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true_np, y_pred_np).ravel()
    accuracy = sklearn.metrics.accuracy_score(y_true_np, y_pred_np)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    iou = tp / (tp + fp + fn)

    print('accuracy: %.4f' % accuracy)
    print('sensitivity: %.4f' % sensitivity)
    print('specificity: %.4f' % specificity)
    print('iou: %.4f' % iou)

