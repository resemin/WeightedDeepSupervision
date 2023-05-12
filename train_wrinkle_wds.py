# Copyright (c) 2023,Semin Kim, AI R&D Center, lululab

import os
import numpy as np
import random
from tqdm import tqdm

import torch
from torch.autograd import Variable
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from models.unet_model import UNet_texture_front_ds
from loss.losses import dice_loss
from dataset.datasets_AIIM import Dataset_Wrinkle_WDS
from utils.utils import add_weight_decay

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

    model_id = "1"
    agum_file = '1'
    num_epochs = 200
    train_batch_size = 4
    val_batch_size = 2
    learning_rate = 0.01

    for step_f in range(0, 6):

        # Edit your path
        path_ref = 'Wrinkle/DB/AIIM/Rev/6_fold/%d' % step_f
        path_src = 'Wrinkle/DB/AIIM/Rev/src'
        path_ttr = 'Wrinkle/DB/AIIM/Rev/texture'
        path_gnd = 'Wrinkle/DB/AIIM/Rev/GT'

        list_src = os.listdir(path_src)

        list_tr = list()
        list_te = list()

        for step_s, str_src in enumerate(list_src):
            fns_ref = '%s/%s' % (path_ref, str_src)

            if os.path.isfile(fns_ref):
                list_te.append(str_src)
            else:
                list_tr.append(str_src)

        train_dataset = Dataset_Wrinkle_WDS(list_src=list_tr, path_src=path_src, path_lbl=path_gnd, path_ttr=path_ttr, b_aug=True)
        val_dataset = Dataset_Wrinkle_WDS(list_src=list_te, path_src=path_src, path_lbl=path_gnd, path_ttr=path_ttr, b_aug=False)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=train_batch_size, shuffle=True,
                                                   num_workers=4)

        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=val_batch_size, shuffle=False,
                                                 num_workers=2)

        str_log = 'WRINKLE_WDS_%d' % step_f
        path_save = 'save_model/%s' % str_log
        if os.path.isdir(path_save) == False:
            os.makedirs((path_save))

        model = UNet_texture_front_ds(4, 2).to(device)

        params = add_weight_decay(model, l2_value=0.0001)
        optimizer = torch.optim.Adam(params, lr=learning_rate)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0.000001)

        writer_loss_tr_1 = SummaryWriter(log_dir='logs/%s/loss_tr_1' % str_log)
        writer_loss_tr_2 = SummaryWriter(log_dir='logs/%s/loss_tr_2' % str_log)
        writer_loss_tr_3 = SummaryWriter(log_dir='logs/%s/loss_tr_3' % str_log)
        writer_loss_tr_4 = SummaryWriter(log_dir='logs/%s/loss_tr_4' % str_log)
        writer_loss_tr = SummaryWriter(log_dir='logs/%s/loss_tr' % str_log)
        writer_loss_te_1 = SummaryWriter(log_dir='logs/%s/loss_te_1' % str_log)
        writer_loss_te_2 = SummaryWriter(log_dir='logs/%s/loss_te_2' % str_log)
        writer_loss_te_3 = SummaryWriter(log_dir='logs/%s/loss_te_3' % str_log)
        writer_loss_te_4 = SummaryWriter(log_dir='logs/%s/loss_te_4' % str_log)
        writer_loss_te = SummaryWriter(log_dir='logs/%s/loss_te' % str_log)
        writer_jsi = SummaryWriter(log_dir='logs/%s/jsi' % str_log)

        epsilon = 2.22045e-16
        jsi_max = 0
        for epoch in range(num_epochs):
            model.train()
            batch_losses = []
            batch_losses_1 = []
            batch_losses_2 = []
            batch_losses_3 = []
            batch_losses_4 = []

            for step, (imgs, label_imgs, img_ttr, img_wds_2, img_wds_3, img_wds_4) in enumerate(tqdm(train_loader)):
                imgs = Variable(imgs).to(device)
                label_imgs = Variable(label_imgs.type(torch.LongTensor)).to(device)
                img_ttr = Variable(img_ttr).to(device)
                img_wds_2 = Variable(img_wds_2.type(torch.FloatTensor)).to(device)
                img_wds_3 = Variable(img_wds_3.type(torch.FloatTensor)).to(device)
                img_wds_4 = Variable(img_wds_4.type(torch.FloatTensor)).to(device)

                out_1, out_2, out_3, out_4 = model(imgs, img_ttr)

                loss_1 = dice_loss(F.softmax(out_1, dim=1).float(), F.one_hot(label_imgs, 2).permute(0, 3, 1, 2).float(), multiclass=True)
                loss_2 = dice_loss(F.softmax(out_2, dim=1).float(), F.one_hot(label_imgs, 2).permute(0, 3, 1, 2).float(), multiclass=True, weight=img_wds_2)
                loss_3 = dice_loss(F.softmax(out_3, dim=1).float(), F.one_hot(label_imgs, 2).permute(0, 3, 1, 2).float(), multiclass=True, weight=img_wds_3)
                loss_4 = dice_loss(F.softmax(out_4, dim=1).float(), F.one_hot(label_imgs, 2).permute(0, 3, 1, 2).float(), multiclass=True, weight=img_wds_4)
                loss = loss_1 + loss_2 + loss_3 + loss_4

                batch_losses_1.append(loss_1.data.cpu().numpy())
                batch_losses_2.append(loss_2.data.cpu().numpy())
                batch_losses_3.append(loss_3.data.cpu().numpy())
                batch_losses_4.append(loss_4.data.cpu().numpy())
                batch_losses.append(loss.data.cpu().numpy())

                optimizer.zero_grad()  # (reset gradients)
                loss.backward()  # (compute gradients)
                optimizer.step()  # (perform optimization step)

            epoch_loss_1 = np.mean(batch_losses_1)
            epoch_loss_2 = np.mean(batch_losses_2)
            epoch_loss_3 = np.mean(batch_losses_3)
            epoch_loss_4 = np.mean(batch_losses_4)
            epoch_loss = np.mean(batch_losses)

            print('epoch %d: train loss_1 = %.08f, lr = %.08f' % (epoch, epoch_loss_1, optimizer.param_groups[0]['lr']))
            scheduler.step()

            writer_loss_tr_1.add_scalar('loss', epoch_loss_1, epoch)
            writer_loss_tr_2.add_scalar('loss', epoch_loss_2, epoch)
            writer_loss_tr_3.add_scalar('loss', epoch_loss_3, epoch)
            writer_loss_tr_4.add_scalar('loss', epoch_loss_4, epoch)
            writer_loss_tr.add_scalar('loss', epoch_loss, epoch)

            model.eval()  # (set in evaluation mode, this affects BatchNorm and dropout)
            batch_losses_1 = []
            batch_losses_3 = []
            batch_losses_2 = []
            batch_losses_4 = []
            batch_losses = []

            y_true = []
            y_score = []
            
            for step, (imgs, label_imgs, img_ttr, img_wds_2, img_wds_3, img_wds_4) in enumerate(tqdm(val_loader)):
                with torch.no_grad():
                    imgs = Variable(imgs).to(device)
                    label_imgs = Variable(label_imgs.type(torch.LongTensor)).to(device)
                    img_ttr = Variable(img_ttr).to(device)
                    img_wds_2 = Variable(img_wds_2.type(torch.FloatTensor)).to(device)
                    img_wds_3 = Variable(img_wds_3.type(torch.FloatTensor)).to(device)
                    img_wds_4 = Variable(img_wds_4.type(torch.FloatTensor)).to(device)

                    out_1, out_2, out_3, out_4 = model(imgs, img_ttr)

                loss_1 = dice_loss(F.softmax(out_1, dim=1).float(), F.one_hot(label_imgs, 2).permute(0, 3, 1, 2).float(), multiclass=True)
                loss_2 = dice_loss(F.softmax(out_2, dim=1).float(), F.one_hot(label_imgs, 2).permute(0, 3, 1, 2).float(), multiclass=True, weight=img_wds_2)
                loss_3 = dice_loss(F.softmax(out_3, dim=1).float(), F.one_hot(label_imgs, 2).permute(0, 3, 1, 2).float(), multiclass=True, weight=img_wds_3)
                loss_4 = dice_loss(F.softmax(out_4, dim=1).float(), F.one_hot(label_imgs, 2).permute(0, 3, 1, 2).float(), multiclass=True, weight=img_wds_4)
                loss = loss_1 + loss_2 + loss_3 + loss_4

                batch_losses_1.append(loss_1.data.cpu().numpy())
                batch_losses_2.append(loss_2.data.cpu().numpy())
                batch_losses_3.append(loss_3.data.cpu().numpy())
                batch_losses_4.append(loss_4.data.cpu().numpy())
                batch_losses.append(loss.data.cpu().numpy())

                score = torch.softmax(out_1, dim=1).cpu().numpy()
                score = score[:, 1, :, :]

                label_imgs_np = label_imgs.cpu().numpy()

                y_score.append(score.reshape(-1))
                y_true.append(label_imgs_np.reshape(-1))

                y_score_np = np.asarray(y_score).reshape(-1)
                y_pred_np = y_score_np > 0.5
                y_true_np = np.asarray(y_true).reshape(-1)

            epoch_loss_1 = np.mean(batch_losses_1)
            epoch_loss_2 = np.mean(batch_losses_2)
            epoch_loss_3 = np.mean(batch_losses_3)
            epoch_loss_4 = np.mean(batch_losses_4)
            epoch_loss = np.mean(batch_losses)

            jsi = metrics.jaccard_score(y_true_np, y_pred_np)            
            epoch_loss = np.mean(batch_losses)
            print('epoch %d: val loss = %.08f' % (epoch, epoch_loss_1))
            print('epoch %d: jsi = %.04f\n' % (epoch, jsi))
            writer_loss_te.add_scalar('loss', epoch_loss, epoch)
            writer_loss_te_1.add_scalar('loss', epoch_loss_1, epoch)
            writer_loss_te_2.add_scalar('loss', epoch_loss_2, epoch)
            writer_loss_te_3.add_scalar('loss', epoch_loss_3, epoch)
            writer_loss_te_4.add_scalar('loss', epoch_loss_4, epoch)
            writer_loss_te.add_scalar('loss', epoch_loss, epoch)

            writer_jsi.add_scalar('jsi', jsi, epoch)           
            
            if jsi_max < jsi:
                jsi_max = jsi
                fns_check = '%s/model_epoch_%d_jsi_%.4f.pth' % (path_save, epoch, jsi)
                torch.save(model.state_dict(), fns_check)                
