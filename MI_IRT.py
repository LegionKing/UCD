'''
MI IRT (Characterizing Sources of Uncertainty in Item Response Theory Scale Scores)
SEM (SEM of another flavour: Two new applications of the supplemented EM algorithm)
'''
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import normal, kl
from torch.distributions.distribution import Distribution
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from EduCDM import CDM
from sklearn.metrics import roc_auc_score, accuracy_score
import logging
import os
from torch.distributions.utils import _standard_normal
import math
import scipy.stats


def mi_irt(user_n, item_n, train_data, valid_data, test_data, quadrature_nodes=18, MI_M=10, device='cpu', skip_value=-1):
    logging.info(f'mi_irt, quadrature_nodes={quadrature_nodes}, MI_M={MI_M}')
    # calculate item parameters
    print('estimating item parameters ...')
    diff_mean, disc_mean, cov_matrix = calculate_item_par(train_data, quadrature_nodes, device)

    # estimate the distribution of students' abilities based on the posterior distributions of item parameters
    print('estimating student parameters')
    # # sample item parameters
    gamma_all_samples = np.empty((item_n, MI_M, 2))
    for i in range(item_n):
        gamma = scipy.stats.multivariate_normal.rvs(mean=[diff_mean[i], disc_mean[i]], cov=cov_matrix[i], size=MI_M)
        gamma_all_samples[i] = gamma
    # # estimate each student's parameters
    quadrature_points, GH_weight, ll_y1_exerstu_ks = common_for_calculate_stu_par(gamma_all_samples, quadrature_nodes=100)
    theta_mean_sampled, theta_var_sampled = np.empty((user_n, MI_M)), np.empty((user_n, MI_M))
    for stu_i in range(user_n):
        if (stu_i + 1) % 100 == 0:
            print(f'student {stu_i + 1}')
        x = [(j + 1, train_data[stu_i][j]) for j in range(item_n) if train_data[stu_i][j] != skip_value]
        for sample_j in range(MI_M):
            theta_mean, theta_var = calculate_stu_par(x, quadrature_points, GH_weight, ll_y1_exerstu_ks[sample_j])
            theta_mean_sampled[stu_i][sample_j] = theta_mean
            theta_var_sampled[stu_i][sample_j] = theta_var
    theta_mean_mean = theta_mean_sampled.mean(axis=1)
    B = 1 / (MI_M - 1) * np.sum((theta_mean_sampled - theta_mean_mean.reshape((user_n, 1))) ** 2, axis=1)
    theta_var_mean = theta_var_sampled.mean(axis=1) + (1 + 1 / MI_M) * B

    # save results
    dst_path = 'result/{}/{}/mi_irt_qn{}mm{}.pars.pkl'.format(data_name, model_name, quadrature_nodes, MI_M)
    print(f'saving results to {dst_path}')
    logging.info(f'saving results to {dst_path}')
    with open(dst_path, 'wb') as o_f:
        pickle.dump({'theta': (theta_mean_mean, theta_var_mean), 'gamma': (diff_mean, disc_mean, cov_matrix)}, o_f)

    # test
    auc, acc = test_model(theta_mean_mean, theta_var_mean, diff_mean, disc_mean, cov_matrix, test_data)
    logging.info(f'auc={auc}, acc={acc}')
    print(f'auc={auc}, acc={acc}')


def irt(stu_par, diff, disc, tensor=True):
    '''
    :param stu_par: (quadrature_nodes,)
    :param diff: (item_n,)
    :param disc: (item_n,)
    :return:
    '''
    if tensor:
        x = (disc * (stu_par.view(-1, 1) - diff)).to(diff.device)  # (quadrature_nodes, item_n)
        mask = (x >= 0)
        ret = torch.zeros(x.shape).double().to(diff.device)
        ret[mask] = 1 / (1 + torch.exp(-x[mask]))
        ret[~mask] = torch.exp(x[~mask]) / (1 + torch.exp(x[~mask]))  # avoid overflow
    else:
        x = disc * (stu_par.reshape(-1, 1) - diff)  # (quadrature_nodes, item_n)
        mask = (x >= 0)
        ret = np.zeros(x.shape)
        ret[mask] = 1 / (1 + np.exp(-x[mask]))
        ret[~mask] = np.exp(x[~mask]) / (1 + np.exp(x[~mask]))  # avoid overflow
    return ret


def __convergence(value_curr, value_next):
    # ret = (abs(value_next - value_curr) > 0.01).sum() == 0
    # loose the covergence condition, otherwise it would be too difficult to converge for large datasets
    ret = (abs(value_next - value_curr) > 0.01).sum() < len(value_next) * 0.1

    return ret


def __single_EM(data_matrix, stu_prior, GH_weight, diff_k, disc_k, device='cpu', skip_value=-1):
    '''
    One iteraction of EM
    :param data_matrix:
    :param stu_prior:
    :param GH_weight:
    :param diff_k:
    :param disc_k:
    :param device:
    :param skip_value:
    :return:
    '''
    data_matrix = data_matrix.to(device)
    stu_prior = stu_prior.to(device)
    diff_k, disc_k = diff_k.to(device), disc_k.to(device)
    user_n, item_n = data_matrix.shape
    quadrature_nodes = len(stu_prior)
    # # E Step
    if Debug: print('E step')
    ll_y1_exerstu_k = irt(stu_prior, diff_k, disc_k)  # (quadrature_nodes, item_n)
    ll_ystu_exer_k = torch.zeros(quadrature_nodes, user_n).to(device)
    mask = data_matrix == skip_value  # without response
    for s in range(quadrature_nodes):
        for i in range(user_n):
            tmp = data_matrix[i] * ll_y1_exerstu_k[s] + (1 - data_matrix[i]) * (1 - data_matrix[i])
            tmp[mask[i]] = 1.
            ll_ystu_exer_k[s][i] = tmp.prod() * GH_weight[s]
    ll_y_exer_k = ll_ystu_exer_k.sum(dim=0)  # (user_n,)
    postp_stu_yexer_k = ll_ystu_exer_k / ll_y_exer_k  # (quadrature_nodes, user_n)
    r_1_k = torch.mm(postp_stu_yexer_k, torch.mul(data_matrix, ~mask))
    r_0_k = torch.mm(postp_stu_yexer_k, torch.mul(1 - data_matrix, ~mask))

    # ## M Step
    diff, disc = nn.Parameter(diff_k.clone()), nn.Parameter(disc_k.clone())
    optimizer = optim.Adam([diff, disc], lr=0.01)
    if Debug: print('M step')
    diff_cur, disc_cur = diff_k.clone(), disc_k.clone()
    while True:
        min_iter = 2
        avg_loss = 0.
        for i in range(min_iter):
            Q = torch.tensor(0.)
            for i in range(quadrature_nodes):
                ll_y1_exerstu = irt(stu_prior, diff, disc)
                Q = (r_1_k * torch.log(ll_y1_exerstu + 1e-10) + r_0_k * torch.log(1 - ll_y1_exerstu + 1e-10)).sum()
            loss = - Q
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        diff_next, disc_next = diff.detach().clone(), disc.detach().clone()
        disc_next = torch.clip(disc_next, 0.001, 4)  # avoid to be negative or too large
        avg_loss /= min_iter
        if Debug:
            print(f'diff_next = {diff_next}, disc_next = {disc_next}')
            print('diff_next - diff_cur = {}, disc_next - disc_cur = {}'.format(diff_next - diff_cur, disc_next - disc_cur))
        if __convergence(diff_cur, diff_next) and __convergence(disc_cur, disc_next):  # end inner iteration
            break
        diff_cur, disc_cur = diff_next, disc_next

    return diff_next.cpu(), disc_next.cpu()


def __calculate_delta_exer_par(data_matrix, stu_prior, GH_weight, diff, disc, device='cpu', skip_value=-1):
    epsilon = 1e-3
    user_n, item_n = data_matrix.shape
    data_matrix = torch.FloatTensor(data_matrix)
    delta_exer_par = torch.empty((item_n, 2, 2))

    diff_next, disc_next = __single_EM(data_matrix, stu_prior, GH_weight, diff, disc, device, skip_value)

    for j in range(item_n):
        diff_eps = diff.clone()
        diff_eps[j] += epsilon
        diff_eps_next, disc_eps_next = __single_EM(data_matrix, stu_prior, GH_weight, diff_eps, disc, device, skip_value)
        delta_exer_par[j][0][0] = (diff_eps_next[j] - diff_next[j]) / epsilon
        delta_exer_par[j][1][0] = (disc_eps_next[j] - disc_next[j]) / epsilon

        disc_eps = disc.clone()
        disc_eps[j] += epsilon
        diff_eps_next, disc_eps_next = __single_EM(data_matrix, stu_prior, GH_weight, diff, disc_eps, device, skip_value)
        delta_exer_par[j][0][1] = (diff_eps_next[j] - diff_next[j]) / epsilon
        delta_exer_par[j][1][1] = (disc_eps_next[j] - disc_next[j]) / epsilon
    return delta_exer_par


def calculate_item_par(data_matrix, quadrature_nodes, device='cpu', skip_value=-1):
    # Gauss-Hermite integration，ability in [-3,3]
    GH_weight = np.empty(quadrature_nodes)
    quadrature_points = np.empty(quadrature_nodes)
    interval = 6 / quadrature_nodes
    for i in range(quadrature_nodes):
        point = -3 + interval / 2 + i * interval
        quadrature_points[i] = point
        GH_weight[i] = scipy.stats.norm.pdf(point)
    GH_weight = GH_weight / GH_weight.sum()
    # initialization
    user_n, item_n = data_matrix.shape
    data_matrix = torch.FloatTensor(data_matrix)
    stu_prior = torch.tensor(quadrature_points)
    diff_k, disc_k = torch.zeros(item_n), torch.ones(item_n),
    min_outer_inter, o_iter = 10, 0
    # EM
    while True:
        print('iter', o_iter)
        diff_kplus1, disc_kplus1 = __single_EM(data_matrix, stu_prior, GH_weight, diff_k, disc_k, device, skip_value)
        o_iter += 1
        if (o_iter > min_outer_inter) and __convergence(diff_k, diff_kplus1) and __convergence(disc_k, disc_kplus1):  # end outer iteration
            break
        else:
            print(abs(diff_k - diff_kplus1), abs(disc_k - disc_kplus1))
            diff_k = diff_kplus1
            disc_k = disc_kplus1
    # calculate information matrix
    ll_y1_exerstu_k = irt(stu_prior, diff_kplus1, disc_kplus1)  # (quadrature_nodes, item_n)
    ll_ystu_exer_k = torch.zeros(quadrature_nodes, user_n)
    mask = data_matrix == skip_value
    for s in range(quadrature_nodes):
        for i in range(user_n):
            tmp = data_matrix[i] * ll_y1_exerstu_k[s] + (1 - data_matrix[i]) * (1 - data_matrix[i])
            tmp[mask[i]] = 1.
            ll_ystu_exer_k[s][i] = tmp.prod() * GH_weight[s]
    ll_y_exer_k = ll_ystu_exer_k.sum(dim=0)  # (user_n,)
    postp_stu_yexer_k = ll_ystu_exer_k / ll_y_exer_k  # (quadrature_nodes, user_n)
    r_1_k = torch.mm(postp_stu_yexer_k, torch.mul(data_matrix, ~mask))    # (quadrature_nodes, item_n)
    r_0_k = torch.mm(postp_stu_yexer_k, torch.mul(1 - data_matrix, ~mask))  # (quadrature_nodes, item_n)

    # save temporary results
    dst_path = 'result/{}/{}/mi_irt_qn{}mm{}.exertemp.pkl'.format(data_name, model_name, quadrature_nodes, 10)
    print(f'saving results to {dst_path}')
    logging.info(f'saving results to {dst_path}')
    with open(dst_path, 'wb') as o_f:
        pickle.dump({'diff': diff_kplus1, 'disc': disc_kplus1, 'r1k': r_1_k, 'r0k': r_0_k, 'll_y1_exerstu_k': ll_y1_exerstu_k}, o_f)

    info_matrix_c = torch.zeros((item_n, 2, 2))
    for j in range(item_n):
        for i in range(quadrature_nodes):
            p_irt, q_irt = ll_y1_exerstu_k[i][j], 1 - ll_y1_exerstu_k[i][j]
            x1, x0 = r_1_k[i][j], r_0_k[i][j]
            info_matrix_c[j][0][0] += (quadrature_points[i] - diff_kplus1[j]) ** 2 * p_irt * q_irt * (x1 + x0)
            info_matrix_c[j][0][1] += -(
                        disc_kplus1[j] * (quadrature_points[i] - diff_kplus1[j]) * p_irt * q_irt * (x1 + x0)
                        + x0 * p_irt - x1 * q_irt)
            info_matrix_c[j][1][1] += disc_kplus1[j] ** 2 * p_irt * (1 - p_irt) * (x1 + x0)
        info_matrix_c[j][1][0] = info_matrix_c[j][0][1]
    # calculate delta_exerpar
    delta_exerpar = __calculate_delta_exer_par(train_data, stu_prior, GH_weight, diff_kplus1, disc_kplus1, device, skip_value)
    info_matrix_o = torch.matmul(torch.eye(2) - delta_exerpar, info_matrix_c)

    cov_matrix = np.zeros(info_matrix_c.shape)
    for j in range(item_n):
        if (diff_kplus1[j] == 0) and (disc_kplus1[j] == 1):  # initialized values，didn't trained，no relavant responses
            continue
        else:
            cov_matrix[j] = np.linalg.inv(info_matrix_o[j].numpy())

    return diff_kplus1.numpy(), disc_kplus1.numpy(), cov_matrix


def irt_single(stu_par, diff, disc, tensor=False):
    '''
    :param stu_par: (1,)
    :param diff: (n,)
    :param disc: (n,)
    :param tensor:
    :return:
    '''
    x = disc * (stu_par - diff)
    mask = (x >= 0)
    if tensor:
        ret = torch.zeros(x.shape)
        ret[mask] = 1 / (1 + torch.exp(-x[mask]))
        ret[~mask] = torch.exp(x[~mask]) / (1 + torch.exp(x[~mask]))   # avoid overflow
    else:
        ret = np.zeros(x.shape)
        ret[mask] = 1 / (1 + np.exp(-x[mask]))
        ret[~mask] = np.exp(x[~mask]) / (1 + np.exp(x[~mask]))
    return ret


def common_for_calculate_stu_par(gamma_all_samples, quadrature_nodes=100):
    GH_weight = np.empty(quadrature_nodes)
    quadrature_points = np.empty(quadrature_nodes)
    interval = 6 / quadrature_nodes
    for i in range(quadrature_nodes):
        point = -3 + interval / 2 + i * interval
        quadrature_points[i] = point
        GH_weight[i] = scipy.stats.norm.pdf(point)
    GH_weight = GH_weight / GH_weight.sum()

    item_n, MI_M, _ = gamma_all_samples.shape
    ll_y1_exerstu_ks = np.empty((MI_M, quadrature_nodes, item_n))
    for i in range(MI_M):
        ll_y1_exerstu_ks[i] = irt(quadrature_points, gamma_all_samples[:, i, 0], gamma_all_samples[:, i, 1], tensor=False)
    ll_y1_exerstu_ks = np.clip(ll_y1_exerstu_ks, 0.0001, 0.9999)
    return quadrature_points, GH_weight, ll_y1_exerstu_ks


def calculate_stu_par(x, quadrature_points, GH_weight, ll_y1_exerstu_k):
    '''
    Calculate the student parameters under the sampled item parameters
    :param x:
    :param diffs:
    :param discs:
    :return:
    '''
    quadrature_nodes = len(quadrature_points)
    ll_ystu_exer_k = np.zeros(quadrature_nodes)
    x = np.array(x)
    item_idx, score = x[:, 0] - 1, x[:, 1]
    item_idx = item_idx.astype(int)
    for point_i in range(quadrature_nodes):
        ll_ystu_exer_k[point_i] = (ll_y1_exerstu_k[point_i][item_idx] * score + (1 - ll_y1_exerstu_k[point_i][item_idx]) * (1 - score)).prod()

    # calculate the mean of theta
    fx_gamma = (ll_ystu_exer_k * GH_weight).sum()
    theta_mean = (quadrature_points * ll_ystu_exer_k * GH_weight).sum() / fx_gamma

    # calculate the variance of theta
    theta_var = (ll_ystu_exer_k * (quadrature_points ** 2) * GH_weight).sum() / fx_gamma - theta_mean ** 2

    return theta_mean, theta_var


def test_model(theta_mean, theta_var, diff_mean, disc_mean, cov_matrix, test_data):
    print('test_model ...')
    logging.info('test_model ...')
    ys = test_data['score'].values
    preds = []
    for _, row in test_data.iterrows():
        user_id, item_id = int(row['user_id']), int(row['item_id'])
        pred = irt_single(theta_mean[user_id - 1], diff_mean[item_id - 1], disc_mean[item_id - 1])
        preds.append(pred)
    return roc_auc_score(ys, preds), accuracy_score(ys, np.array(preds) >= 0.5)


def PICP_pred(pred_up, pred_down, y):
    '''
    :param pred_up: shape: (sample_n, batch_size)
    :param pred_down: shape: (sample_n, batch_size)
    :param y: shape: (sample_n, batch_size)
    :return:
    '''
    sample_n, batch_size = pred_up.shape
    assert (pred_up.shape == pred_down.shape) and (batch_size == len(y))
    l = abs(y - pred_down) < 0.5
    r = abs(y - pred_up) < 0.5
    cover_cnt = np.array(l | r, dtype=float).mean(axis=0).sum()
    return cover_cnt


def PINAW(up, down, range_y=1, reduce=False):
    '''
    :param pred_up: shape: (sample_n, batch_size, *)
    :param pred_down: shape: (sample_n, batch_size, *)
    :param range_y:
    :param reduce:
    :return:
    '''
    assert up.shape == down.shape
    a = (up - down).mean(axis=0) / range_y
    if reduce:
        a = a.mean()
    return a


def CWC(picp, pinaw, mu=0.95, eta=10):
    gamma = 1 if picp < mu else 0
    return pinaw * (1 + gamma * np.exp(eta * (mu - picp)))


def calculate_interval_metric(data_set, args, d, mu, eta):
    logging.info(f'calculate_interval_metric ... d={d}, mu={mu}, eta={eta}')
    dst_path = 'result/{}/{}/mi_irt_qn{}mm{}.pars.pkl'.format(data_name, model_name, args.qnodes, args.mi_m)
    logging.info(f'loading model from {dst_path}')
    with open(dst_path, 'rb') as i_f:
        pars = pickle.load(i_f)
    theta_mean_mean, theta_var_mean = pars['theta']
    diff_mean, disc_mean, cov_matrix = pars['gamma']

    ys = data_set['score'].values
    preds_up_sampled = np.empty((len(ys), args.sample_n))
    preds_down_sampled = np.empty((len(ys), args.sample_n))
    idx = -1
    for _, row in data_set.iterrows():
        idx += 1
        user_id, item_id = int(row['user_id']), int(row['item_id'])
        theta_up = theta_mean_mean[user_id - 1] + d * np.sqrt(theta_var_mean[user_id - 1])
        theta_down = theta_mean_mean[user_id - 1] - d * np.sqrt(theta_var_mean[user_id - 1])
        diff_mean_i, disc_mean_i, cov_matrix_i = diff_mean[item_id - 1], disc_mean[item_id - 1], cov_matrix[item_id - 1]
        gamma_sampled = scipy.stats.multivariate_normal.rvs(mean=[diff_mean_i, disc_mean_i], cov=cov_matrix_i, size=args.sample_n)
        preds_up_sampled[idx] = irt_single(theta_up, gamma_sampled[:, 0], gamma_sampled[:, 1])
        preds_down_sampled[idx] = irt_single(theta_down, gamma_sampled[:, 0], gamma_sampled[:, 1])
    preds_up_sampled = preds_up_sampled.transpose()
    preds_down_sampled = preds_down_sampled.transpose()
    picp = PICP_pred(preds_up_sampled, preds_down_sampled, ys) / len(ys)
    pinaw = PINAW(preds_up_sampled, preds_down_sampled).mean()
    cwc = CWC(picp, pinaw, mu, eta)
    logging.info(f'PICP={picp}, PINAW={pinaw}, CWC={cwc}')
    print(f'PICP={picp}, PINAW={pinaw}, CWC={cwc}')


def load_data(data_name):
    df_train = pd.read_csv("data/{}/train.csv".format(data_name))
    df_valid = pd.read_csv("data/{}/valid.csv".format(data_name))
    df_test = pd.read_csv("data/{}/test.csv".format(data_name))

    user_n = np.max(df_train['user_id'])
    item_n = np.max([np.max(df_train['item_id']), np.max(df_valid['item_id']), np.max(df_test['item_id'])])

    train_data = - np.ones((user_n, item_n))  # skip value == -1
    for row in df_train.itertuples():
        train_data[row.user_id - 1][row.item_id - 1] = row.score

    return user_n, item_n, train_data, df_valid, df_test


def test(data_name):
    user_n, item_n, train_data, valid_data, test_data = load_data(data_name)
    dst_path = 'result/{}/{}/mi_irt_qn{}mm{}.pars.pkl'.format(data_name, model_name, 18, 10)
    with open(dst_path, 'rb') as i_f:
        pars = pickle.load(i_f)
    theta_mean_mean, theta_var_mean = pars['theta']
    diff_mean, disc_mean, cov_matrix = pars['gamma']
    auc, acc = test_model(theta_mean_mean, theta_var_mean, diff_mean, disc_mean, cov_matrix, train_data)
    logging.info(f'auc={auc}, acc={acc}')
    print(f'auc={auc}, acc={acc}')


if __name__ == '__main__':
    model_name = 'MI_IRT'
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='FrcSub', help='data name')
    parser.add_argument('--qnodes', type=int, default=100, help='quadrature nodes')
    parser.add_argument('--mi-m', type=int, default=10, help='MI_M')
    parser.add_argument('--sample-n', type=int, default=30, help='sample_n')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--debug', type=int, default=0)
    args = parser.parse_args()

    Debug = args.debug != 0

    data_name = args.data
    dst_folder = 'result/{}/{}'.format(data_name, model_name)
    if not os.path.isdir(dst_folder):
        os.makedirs(dst_folder)
    logging.basicConfig(
        filename='{}/{}-qn{}mm{}.log'.format(dst_folder, model_name, args.qnodes, args.mi_m),
        level=logging.INFO, format='%(asctime)s %(message)s')

    user_n, item_n, train_data, valid_data, test_data = load_data(data_name)
    mi_irt(user_n, item_n, train_data, valid_data, test_data, args.qnodes, args.mi_m, args.device, skip_value=-1)
    calculate_interval_metric(test_data, args, d=1.96, mu=0.95, eta=10)

