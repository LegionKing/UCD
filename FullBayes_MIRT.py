'''
Run with pystan 3.6 (on linux）
'''

import pandas as pd
import numpy as np
# import pystan
import stan
import pickle
import os
import argparse
import logging
from sklearn.metrics import roc_auc_score, accuracy_score


def full_bayes_train(data, fix_sigma=False, sigma_diff=1., sigma_disc=1., latent_dim=10):
    if fix_sigma:
        code = '''
            data {
            int<lower=1> J;                     // number of students
            int<lower=1> K;                     // number of questions
            int<lower=1> N;                     // number of observations
            array[N] int<lower=1, upper=J> jj;  // student for observation n
            array[N] int<lower=1, upper=K> kk;  // question for observation n
            array[N] int<lower=0, upper=1> y;   // correctness for observation n
            }

            parameters {
            array[J] vector[%d] theta;             // ability for j - mean
            array[K] real diff;              // difficulty for k
            array[K] vector<lower=0, upper=1>[%d] disc;    // discrimination of k
            }

            model {
            for (n in 1:J) {
                theta[n] ~ std_normal();
            }
            for (n in 1:K) {
                disc[n] ~ lognormal(0, %f);
            }
            diff ~ normal(0, %f);
            for (n in 1:N) {
                y[n] ~ bernoulli_logit((disc[kk[n]]' * theta[jj[n]]) - diff[kk[n]]);
            }
            }
            ''' % (latent_dim, latent_dim, sigma_disc, sigma_diff)
    else:
        code = '''
            data {
            int<lower=1> J;                     // number of students
            int<lower=1> K;                     // number of questions
            int<lower=1> N;                     // number of observations
            array[N] int<lower=1, upper=J> jj;  // student for observation n
            array[N] int<lower=1, upper=K> kk;  // question for observation n
            array[N] int<lower=0, upper=1> y;   // correctness for observation n
            }

            parameters {
            array[J] vector[%d] theta;             // ability for j - mean
            array[K] real diff;              // difficulty for k
            array[K] vector<lower=0>[10] disc;    // discrimination of k
            array[K] real<lower=0.01> sigma_diff;    // scale of difficulties
            array[K] vector<lower=0.01, upper=1>[%d] sigma_disc;   // scale of log discrimination
            }

            model {
            for (n in 1:J) {
                theta[n] ~ std_normal();
            }
            sigma_diff ~ cauchy(0, 5);
            sigma_disc ~ cauchy(0, 5);
            diff ~ normal(0, sigma_diff);
            for (n in 1:K) {
                disc[n] ~ lognormal(0, sigma_disc[n]);
            }
            for (n in 1:N) {
                y[n] ~ bernoulli_logit((disc[kk[n]]' * theta[jj[n]]) - diff[kk[n]]);
            }
            }
            ''' % (latent_dim, latent_dim)

    # pystan 2.xx
    # sm = pystan.StanModel(model_code=code)
    # fit = sm.sampling(data=data, iter=1000, chains=1)
    # pars = fit.extract(permuted=True, inc_warmup=False)  # 扔掉warm up的采样

    # pystan 3.xx
    logging.warning('full_bayes_train() ...')
    sm = stan.build(code, data=data)
    pars = sm.sample(num_chains=1, num_samples=500, num_warmup=1000)

    # save model
    dst_folder = 'result/{}/{}/{}_fs{}sdf{}sdc{}ld{}.pkl'.format(data_name, model_name, model_name, int(fix_sigma), sigma_diff, sigma_disc, latent_dim)
    logging.warning('saving model parameters to {}'.format(dst_folder))
    with open(dst_folder, 'wb') as o_f:
        pickle.dump(pars, o_f)

    # calculate the means and variations
    theta_mean = pars['theta'].mean(axis=1)
    theta_std = pars['theta'].std(axis=1)
    diff_mean = pars['diff'].mean(axis=1)
    diff_std = pars['diff'].std(axis=1)
    disc_mean = pars['disc'].mean(axis=1)
    disc_std = pars['disc'].std(axis=1)

    return (theta_mean, theta_std), (diff_mean, diff_std), (disc_mean, disc_std)


def load_data(data_name):
    def transform(df, user_n, item_n):
        data = {
            'J': user_n,
            'K': item_n,
            'N': len(df),
            'jj': df['user_id'].values.astype(int),
            'kk': df['item_id'].values.astype(int),
            'y': df['score'].values.astype(int)
        }
        return data

    train_data = pd.read_csv("data/{}/train.csv".format(data_name))
    valid_data = pd.read_csv("data/{}/valid.csv".format(data_name))
    test_data = pd.read_csv("data/{}/test.csv".format(data_name))

    user_n = np.max(train_data['user_id'])
    item_n = np.max([np.max(train_data['item_id']), np.max(valid_data['item_id']), np.max(test_data['item_id'])])
    train_set, valid_set, test_set = [transform(data_set, user_n, item_n) for data_set in [train_data, valid_data, test_data]]
    return user_n, item_n, train_set, valid_set, test_set


def test(data_set, theta, diff, disc, args):
    logging.warning('test ...')
    ys = data_set['y']
    user_ids = data_set['jj']
    item_ids = data_set['kk']
    preds = []
    for i in range(len(ys)):
        user_id = user_ids[i]
        item_id = item_ids[i]
        preds.append(mirt(theta[0][user_id - 1], diff[0][item_id - 1], disc[0][item_id - 1]))

    auc = roc_auc_score(ys, preds)
    acc = accuracy_score(ys, np.array(preds) >= 0.5)
    logging.warning(f'auc={auc}, acc={acc}')
    print(f'auc={auc}, acc={acc}')


def test_mc(data_set, theta, diff, disc, args):
    '''
    The input theta, diff, and disc are samples from distributions
    :param data_set:
    :param theta:
    :param diff:
    :param disc:
    :param args:
    :return:
    '''
    logging.warning('test_mc ...')
    ys = data_set['y']
    user_ids = data_set['jj']
    item_ids = data_set['kk']
    preds = []
    for i in range(len(ys)):
        user_id = user_ids[i]
        item_id = item_ids[i]
        preds.append(mirt(theta[user_id - 1], diff[item_id - 1], disc[item_id - 1]).mean())

    auc = roc_auc_score(ys, preds)
    acc = accuracy_score(ys, np.array(preds) >= 0.5)
    logging.warning(f'auc={auc}, acc={acc}')
    print(f'auc={auc}, acc={acc}')


def calculate_interval_metric(data_set, args, d, mu, eta, percentage):
    logging.warning('calculate_interval_metric ... d={}, mu={}, eta={}'.format(d, mu, eta))
    ys = data_set['y']
    user_ids = data_set['jj']
    item_ids = data_set['kk']
    preds_up, preds_down = [], []
    for i in range(len(ys)):
        user_id = user_ids[i]
        item_id = item_ids[i]
        theta_up = theta[0][user_id - 1] + d * theta[1][user_id - 1]
        theta_down = theta[0][user_id - 1] - d * theta[1][user_id - 1]
        preds_up.append(mirt(theta_up, diff[0][item_id - 1], disc[0][item_id - 1]))
        preds_down.append(mirt(theta_down, diff[0][item_id - 1], disc[0][item_id - 1]))
    preds_up = np.array(preds_up).reshape(1, -1)
    preds_down = np.array(preds_down).reshape(1, -1)
    picp = PICP_pred(preds_up, preds_down, ys, percentage) / len(ys)
    pinaw = PINAW(preds_up, preds_down, reduce=True)
    cwc = CWC(picp, pinaw, mu, eta)
    print('PICP={}, PINAW={}, CWC={}'.format(picp, pinaw, cwc))
    logging.warning('PICP={}, PINAW={}, CWC={}'.format(picp, pinaw, cwc))


def mirt(stu_par, diff, disc):
    return 1 / (1 + np.exp(- (disc * stu_par).sum() + diff))


def PICP_pred(pred_up, pred_down, y, percentage=False):
    '''
    :param pred_up: shape: (sample_n, batch_size)
    :param pred_down: shape: (sample_n, batch_size)
    :param y: shape: (sample_n, batch_size)
    :return:
    '''
    sample_n, batch_size = pred_up.shape
    assert (pred_up.shape == pred_down.shape) and (batch_size == len(y))
    if not percentage:
        l = abs(y - pred_down) < 0.5
        r = abs(y - pred_up) < 0.5
        cover_cnt = np.array(l | r, dtype=float).mean(axis=0).sum()
    else:
        percent_1 = np.zeros((sample_n, batch_size))
        for i in range(sample_n):
            for j in range(batch_size):
                if pred_down[i][j] >= 0.5:
                    percent_1[i][j] = 1
                elif pred_up[i][j] > 0.5:
                    percent_1[i][j] = (pred_up[i][j] - 0.5) / (pred_up[i][j] - pred_down[i][j])
        cover_cnt = (percent_1 * y + (1 - y) * (1 - percent_1)).mean(axis=0).sum()

    return cover_cnt


def PINAW(up, down, range_y=1, reduce=False):
    '''

    :param pred_up: shape: (sample_n, batch_size, *)
    :param pred_down: shape: (sample_n, batch_size, *)
    :param range_y: domain of definition
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


if __name__ == '__main__':
    model_name = 'FullBayes_MIRT'

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='data name')
    parser.add_argument('--fix-sigma', type=bool, default=False)
    parser.add_argument('--sigma-diff', type=float, default=1.)
    parser.add_argument('--sigma-disc', type=float, default=1.)
    parser.add_argument('--latent-dim', type=int, default=10)
    args = parser.parse_args()

    data_name = args.data
    dst_folder = 'result/{}/{}'.format(data_name, model_name)
    if not os.path.isdir(dst_folder):
        os.makedirs(dst_folder)
    logging.basicConfig(
        filename='{}/{}-fs{}sdf{}sdc{}ld{}.log'.format(dst_folder, model_name, int(args.fix_sigma), args.sigma_diff, args.sigma_disc, args.latent_dim),
        level=logging.WARNING, format='%(asctime)s %(message)s')

    user_n, item_n, train_data, valid_data, test_data = load_data(data_name)
    theta, diff, disc = full_bayes_train(train_data, fix_sigma=args.fix_sigma, sigma_diff=args.sigma_diff, sigma_disc=args.sigma_disc, latent_dim=args.latent_dim)

    path = 'result/{}/{}/{}_fs{}sdf{}sdc{}ld{}.pkl'.format(data_name, model_name, model_name, int(args.fix_sigma), args.sigma_diff, args.sigma_disc, args.latent_dim)
    with open(path, 'rb') as i_f:
        pars = pickle.load(i_f)
    theta_mean = pars['theta'].mean(axis=-1)
    theta_std = pars['theta'].std(axis=-1)
    print(theta_mean, theta_std)
    exit()
    diff_mean = pars['diff'].mean(axis=-1)
    diff_std = pars['diff'].std(axis=-1)
    disc_mean = pars['disc'].mean(axis=-1)
    disc_std = pars['disc'].std(axis=-1)
    theta, diff, disc = (theta_mean, theta_std), (diff_mean, diff_std), (disc_mean, disc_std)
    # test(test_data, theta, diff, disc, args)
    calculate_interval_metric(test_data, args, d=1.96, mu=0.95, eta=10, percentage=False)

    # path = 'result/{}/{}/{}_fs{}sdf{}sdc{}ld{}.pkl'.format(data_name, model_name, model_name, int(args.fix_sigma), args.sigma_diff, args.sigma_disc, args.latent_dim)
    # with open(path, 'rb') as i_f:
    #     pars = pickle.load(i_f)
    # theta, diff, disc = pars['theta'], pars['diff'], pars['disc']
    # print(pars['theta'].std(axis=1) * 3.92, pars['diff'].std(axis=1) * 3.92, pars['disc'].std(axis=1) * 3.92)
    # test_mc(test_data, theta, diff, disc, args)
