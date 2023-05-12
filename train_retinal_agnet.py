# Copyright (c) 2023,Semin Kim, AI R&D Center, lululab

import os
import random
import numpy as np
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from core.models import AG_Net
from dataset.datasets_DRIVE import Dataset_album_DRIVE_AIIM_wds_strong_aug
from loss.losses import dice_loss
from utils.utils import add_weight_decay
from sklearn import metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed = 43
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

    len_h = 512
    len_w = 512

    path_tr_src = '/mnt/hdd1/db/Retinal/DRIVE/training/images_rename_png_584_584'
    path_tr_lbl = '/mnt/hdd1/db/Retinal/DRIVE/training/1st_manual_rename_png_584_584_wds'

    path_va_src = '/mnt/hdd1/db/Retinal/DRIVE/test/images_rename_png_584_584'
    path_va_lbl = '/mnt/hdd1/db/Retinal/DRIVE/test/1st_manual_rename_png_584_584_wds'

    max_pixel = 255
    str_save = 'Rev_new_E200'

    train_dataset = Dataset_album_DRIVE_AIIM_wds_strong_aug(path_src=path_tr_src, path_lbl=path_tr_lbl, b_aug=True, max_pixel=max_pixel)
    val_dataset = Dataset_album_DRIVE_AIIM_wds_strong_aug(path_src=path_va_src, path_lbl=path_va_lbl, b_aug=False, max_pixel=max_pixel)
    path_save = 'save_model/%s' % str_save
    if os.path.isdir(path_save) == False:
        os.makedirs(path_save)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=train_batch_size, shuffle=True,
                                               num_workers=4)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=val_batch_size, shuffle=False,
                                               num_workers=2)

    unet = AG_Net(n_classes=num_class, bn=True, BatchNorm=False).to(device)
    params = add_weight_decay(unet, l2_value=0.0001)
    optimizer = torch.optim.Adam(params, learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.000001)

    writer_loss_tr_all = SummaryWriter(log_dir='logs/%s/loss_tr_all' % str_save)
    writer_loss_tr_5 = SummaryWriter(log_dir='logs/%s/loss_tr_5' % str_save)
    writer_loss_tr_6 = SummaryWriter(log_dir='logs/%s/loss_tr_6' % str_save)
    writer_loss_tr_7 = SummaryWriter(log_dir='logs/%s/loss_tr_7' % str_save)
    writer_loss_tr_8 = SummaryWriter(log_dir='logs/%s/loss_tr_8' % str_save)

    writer_loss_te_all = SummaryWriter(log_dir='logs/%s/loss_te_all' % str_save)
    writer_loss_te_5 = SummaryWriter(log_dir='logs/%s/loss_te_5' % str_save)
    writer_loss_te_6 = SummaryWriter(log_dir='logs/%s/loss_te_6' % str_save)
    writer_loss_te_7 = SummaryWriter(log_dir='logs/%s/loss_te_7' % str_save)
    writer_loss_te_8 = SummaryWriter(log_dir='logs/%s/loss_te_8' % str_save)

    writer_iou = SummaryWriter(log_dir='logs/%s/iou' % str_save)

    iou_max = 0
    softmax_2d = torch.nn.Softmax2d()

    for epoch in range(num_epochs):
        # if (epoch % 10 == 0) and epoch != 0 and epoch < 400:
        #     learning_rate /= 10
        #     optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)

        # Phase: train
        unet.train()
        batch_losses_all = []
        batch_losses_5 = []
        batch_losses_6 = []
        batch_losses_7 = []
        batch_losses_8 = []

        for step, (imgs, label_imgs, wd2, wd3, wd4) in enumerate(tqdm(train_loader)):
            imgs = Variable(imgs).to(device)
            label_imgs = Variable(label_imgs.type(torch.LongTensor)).to(device)
            wd2 = Variable(wd2.type(torch.FloatTensor)).to(device)
            wd3 = Variable(wd3.type(torch.FloatTensor)).to(device)
            wd4 = Variable(wd4.type(torch.FloatTensor)).to(device)
            # outputs = unet(imgs)
            [side_5, side_6, side_7, side_8] = unet(imgs)

            label_imgs_2 = torch.nn.functional.one_hot(label_imgs, 2).permute(0, 3, 1, 2).float()
            loss_5 = dice_loss(softmax_2d(side_5), label_imgs_2, multiclass=True, weight=wd4)
            loss_6 = dice_loss(softmax_2d(side_6), label_imgs_2, multiclass=True, weight=wd3)
            loss_7 = dice_loss(softmax_2d(side_7), label_imgs_2, multiclass=True, weight=wd2)
            loss_8 = dice_loss(softmax_2d(side_8), label_imgs_2, multiclass=True)

            loss_all = loss_5 + loss_6 + loss_7 + loss_8

            loss_value_all = loss_all.data.cpu().numpy()
            loss_value_5 = loss_5.data.cpu().numpy()
            loss_value_6 = loss_6.data.cpu().numpy()
            loss_value_7 = loss_7.data.cpu().numpy()
            loss_value_8 = loss_8.data.cpu().numpy()

            batch_losses_all.append(loss_value_all)
            batch_losses_5.append(loss_value_5)
            batch_losses_6.append(loss_value_6)
            batch_losses_7.append(loss_value_7)
            batch_losses_8.append(loss_value_8)

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            # break

        epoch_loss_all = np.mean(batch_losses_all)
        epoch_loss_5 = np.mean(batch_losses_5)
        epoch_loss_6 = np.mean(batch_losses_6)
        epoch_loss_7 = np.mean(batch_losses_7)
        epoch_loss_8 = np.mean(batch_losses_8)

        print('epoch %d: train loss = %.4f, lr = %f' % (epoch, epoch_loss_8, optimizer.param_groups[0]['lr']))
        scheduler.step()
        writer_loss_tr_all.add_scalar('loss', epoch_loss_all, epoch)
        writer_loss_tr_5.add_scalar('loss', epoch_loss_5, epoch)
        writer_loss_tr_6.add_scalar('loss', epoch_loss_6, epoch)
        writer_loss_tr_7.add_scalar('loss', epoch_loss_7, epoch)
        writer_loss_tr_8.add_scalar('loss', epoch_loss_8, epoch)

        # Phase: evaluation
        unet.eval()
        batch_losses_all = []
        batch_losses_5 = []
        batch_losses_6 = []
        batch_losses_7 = []
        batch_losses_8 = []

        y_true = []
        y_score = []

        for step, (imgs, label_imgs, wd2, wd3, wd4) in enumerate(tqdm(val_loader)):
            with torch.no_grad():
                imgs = Variable(imgs).to(device)
                label_imgs = Variable(label_imgs.type(torch.LongTensor)).to(device)
                wd2 = Variable(wd2.type(torch.FloatTensor)).to(device)
                wd3 = Variable(wd3.type(torch.FloatTensor)).to(device)
                wd4 = Variable(wd4.type(torch.FloatTensor)).to(device)

                [side_5, side_6, side_7, side_8] = unet(imgs)

            label_imgs_2 = torch.nn.functional.one_hot(label_imgs, 2).permute(0, 3, 1, 2).float()

            loss_5 = dice_loss(softmax_2d(side_5), label_imgs_2, multiclass=True, weight=wd4)
            loss_6 = dice_loss(softmax_2d(side_6), label_imgs_2, multiclass=True, weight=wd3)
            loss_7 = dice_loss(softmax_2d(side_7), label_imgs_2, multiclass=True, weight=wd2)
            loss_8 = dice_loss(softmax_2d(side_8), label_imgs_2, multiclass=True)

            loss_all = loss_5 + loss_6 + loss_7 + loss_8

            loss_value_all = loss_all.data.cpu().numpy()
            loss_value_5 = loss_5.data.cpu().numpy()
            loss_value_6 = loss_6.data.cpu().numpy()
            loss_value_7 = loss_7.data.cpu().numpy()
            loss_value_8 = loss_8.data.cpu().numpy()

            batch_losses_all.append(loss_value_all)
            batch_losses_5.append(loss_value_5)
            batch_losses_6.append(loss_value_6)
            batch_losses_7.append(loss_value_7)
            batch_losses_8.append(loss_value_8)

            score = torch.softmax(side_8, dim=1).cpu().numpy()
            score = score[:, 1, :, :]

            label_imgs_np = label_imgs.cpu().numpy()

            y_score.append(score.reshape(-1))
            y_true.append(label_imgs_np.reshape(-1))

            y_score_np = np.asarray(y_score).reshape(-1)
            y_pred_np = y_score_np > 0.5
            y_true_np = np.asarray(y_true).reshape(-1)

        epoch_loss_all = np.mean(batch_losses_all)
        epoch_loss_5 = np.mean(batch_losses_5)
        epoch_loss_6 = np.mean(batch_losses_6)
        epoch_loss_7 = np.mean(batch_losses_7)
        epoch_loss_8 = np.mean(batch_losses_8)

        writer_loss_te_all.add_scalar('loss', epoch_loss_all, epoch)
        writer_loss_te_5.add_scalar('loss', epoch_loss_5, epoch)
        writer_loss_te_6.add_scalar('loss', epoch_loss_6, epoch)
        writer_loss_te_7.add_scalar('loss', epoch_loss_7, epoch)
        writer_loss_te_8.add_scalar('loss', epoch_loss_8, epoch)
        print('epoch %d: val loss = %.4f' % (epoch, epoch_loss_8))

        tn, fp, fn, tp = metrics.confusion_matrix(y_true_np, y_pred_np).ravel()
        accuracy = metrics.accuracy_score(y_true_np, y_pred_np)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        iou = tp / (tp + fp + fn)

        print('epoch %d: iou = %.04f\n' % (epoch, iou))

        writer_iou.add_scalar('cIOU', iou, epoch)

        # Save the best model
        if iou_max < iou:
            iou_max = iou
            fns_check = '%s/unet_epoch_%d_IoU_%.4f.pth' % (path_save, epoch, iou)
            torch.save(unet.state_dict(), fns_check)
