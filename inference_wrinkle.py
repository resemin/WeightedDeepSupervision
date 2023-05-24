# Copyright (c) 2023,Semin Kim, AI R&D Center, lululab

import torch
from torch.autograd import Variable
from models.unet_model import UNet_texture_front_ds
from dataset.datasets_AIIM import Dataset_Wrinkle_WDS
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import sklearn

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device :', device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())

    val_batch_size = 2
    num_class = 2

    # Edit your path
    path_ref = 'Wrinkle/DB/AIIM/Rev/6_fold/0'
    path_src = 'Wrinkle/DB/AIIM/Rev/src'
    path_ttr = 'Wrinkle/DB/AIIM/Rev/texture'
    path_gnd = 'Wrinkle/DB/AIIM/Rev/GT'

    list_src = os.listdir(path_src)
    list_te = list()

    for step_s, str_src in enumerate(list_src):
        fns_ref = '%s/%s' % (path_ref, str_src)
        if os.path.isfile(fns_ref):
            list_te.append(str_src)

    val_dataset = Dataset_Wrinkle_WDS(list_src=list_te, path_src=path_src, path_lbl=path_gnd, path_ttr=path_ttr, b_aug=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=2)

    model = UNet_texture_front_ds(4, 2).to(device)
    fns_mdl = 'save_model/WRINKLE_WDS_0/model_epoch_184_jsi_0.4435.pth'
    model.load_state_dict(torch.load(fns_mdl), strict=True)

    softmax_2d = torch.nn.Softmax2d()

    model.eval()
    EPS = 1e-12

    y_true = []
    y_score = []

    for step, (imgs, label_imgs, img_ttr, img_wds_2, img_wds_3, img_wds_4) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            imgs = Variable(imgs).to(device)
            label_imgs = Variable(label_imgs.type(torch.LongTensor)).to(device)
            img_ttr = Variable(img_ttr).to(device)
            out_1, out_2, out_3, out_4 = model(imgs, img_ttr)

        out = torch.log(softmax_2d(out_1) + EPS)

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
    jsi = sklearn.metrics.jaccard_score(y_true_np, y_pred_np)

    print('jsi: %.4f' % jsi)
    print('accuracy: %.4f' % accuracy)
    print('sensitivity: %.4f' % sensitivity)
    print('specificity: %.4f' % specificity)



    sm = 10
