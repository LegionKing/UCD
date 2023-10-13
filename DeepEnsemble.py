import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import normal, kl
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from EduCDM import CDM
from sklearn.metrics import roc_auc_score, accuracy_score
import logging
import os


def run_trials(args):
    model_name = args.model
    data_name = args.data
    if model_name == 'NCDM':
        from NCDM import NCDM as Model
        from NCDM import load_data
    elif model_name == 'KaNCD':
        from KaNCD import KaNCD as Model
        from KaNCD import load_data
    else:
        print('invalid model name')
        exit()

    stu_n, exer_n, knowledge_n, train_set, valid_set, test_set = load_data(data_name)
    for t_i in range(args.trial):
        print('=============> Trial', t_i)
        dst_folder = 'result/{}/ensemble/{}-trial{}'.format(data_name, model_name, t_i)
        if not os.path.isdir(dst_folder):
            os.makedirs(dst_folder)
        if model_name == 'NCDM':
            logging.basicConfig(filename='{}/{}-lr{}.log'.format(dst_folder, model_name, args.lr),
                                level=logging.INFO, format='%(asctime)s %(message)s')
            model = Model(exer_n=exer_n, student_n=stu_n, knowledge_n=knowledge_n, stu_info_summary=None,
                          exer_info_summary=None, kn_info_summary=None)
            model.train(train_set, valid_set, test_set, data_name, dst_folder, model_name=model_name,
                        device=args.device, epoch_n=args.epoch, lr=args.lr)
        elif model_name == 'KaNCD':
            logging.basicConfig(filename='{}/{}-lr{}dim{}mf{}.log'.format(dst_folder, model_name, args.lr, args.dim, args.mf),
                level=logging.INFO, format='%(asctime)s %(message)s')
            model = Model(exer_n=exer_n, student_n=stu_n, knowledge_n=knowledge_n, mf_type=args.mf, dim=args.dim)
            model.train(train_set, valid_set, test_set, data_name, dst_folder, model_name=model_name, mf_type=args.mf,
                        device=args.device, epoch_n=args.epoch, lr=args.lr)


def eval_average_prediction(args):
    model_name = args.model
    data_name = args.data
    if model_name == 'NCDM':
        from NCDM import NCDM as Model
        from NCDM import load_data
    elif model_name == 'KaNCD':
        from KaNCD import KaNCD as Model
        from KaNCD import load_data
    else:
        print('invalid model name')
        exit()

    stu_n, exer_n, knowledge_n, train_set, valid_set, test_set = load_data(data_name)
    trial_preds = []
    for t_i in range(args.trial):
        dst_folder = 'result/{}/ensemble/{}-trial{}'.format(data_name, model_name, t_i)
        if not os.path.isdir(dst_folder):
            os.makedirs(dst_folder)
        if model_name == 'NCDM':
            logging.basicConfig(filename='{}/{}-lr{}.log'.format(dst_folder, model_name, args.lr),
                                level=logging.INFO, format='%(asctime)s %(message)s')
            model = Model(exer_n=exer_n, student_n=stu_n, knowledge_n=knowledge_n, stu_info_summary=None,
                          exer_info_summary=None, kn_info_summary=None)
            model.load(os.path.join(dst_folder, '{}-lr{}.snapshot{}'.format(model_name, args.lr, args.epoch)))
        elif model_name == 'KaNCD':
            logging.basicConfig(
                filename='{}/{}-lr{}dim{}mf{}.log'.format(dst_folder, model_name, args.lr, args.dim, args.mf),
                level=logging.INFO, format='%(asctime)s %(message)s')
            model = Model(exer_n=exer_n, student_n=stu_n, knowledge_n=knowledge_n, mf_type=args.mf, dim=args.dim)
            model.load(os.path.join(dst_folder, '{}-lr{}dim{}mf{}.snapshot{}'.format(model_name, args.lr, args.dim, args.mf, args.epoch)))

        y_label, y_pred = model.eval_prob(test_set, device=args.device)
        trial_preds.append(y_pred)
    print(np.array(trial_preds).shape, np.array(trial_preds).mean(axis=0).shape)
    trial_preds = np.array(trial_preds)
    pred_ensemble = trial_preds.mean(axis=0)
    auc, acc = roc_auc_score(y_label, pred_ensemble), accuracy_score(y_label, pred_ensemble >= 0.5)
    std = trial_preds.std(axis=0)
    picp = PICP(pred_ensemble, std, y_label)
    pinaw = PINAW(std, alpha=1.96)
    logging.info('eval_average_prediction: AUC={}, ACC={}, PICP={}, PINAW={}'.format(auc, acc, picp, pinaw))
    print('eval_average_prediction: AUC={}, ACC={}, PICP={}, PINAW={}'.format(auc, acc, picp, pinaw))


def PICP(pred_mean, pred_std, y_label, alpha=1.96):
    pred_up = pred_mean + pred_std * alpha
    pred_down = pred_mean - pred_std * alpha
    cover_cnt = 0
    for y, up, down in zip(y_label, pred_up, pred_down):
        if down >= 0.5:
            if y == 1:
                cover_cnt += 1
        elif up <= 0.5:
            if y == 0:
                cover_cnt += 1
        else:
            cover_cnt += 1
    picp = cover_cnt / len(y_label)
    return picp


def PINAW(pred_std, alpha=1.96):
    return np.mean(pred_std * 2 * alpha)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model name')   # NCDM, KaNCD
    parser.add_argument('--data', type=str, default='FrcSub', help='data name')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--trial', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--dim', type=int, default=40)  # KaNCD, 40
    parser.add_argument('--mf', type=str, default='gmf')  # KaNCD, gmf

    args = parser.parse_args()

    run_trials(args)
    eval_average_prediction(args)


