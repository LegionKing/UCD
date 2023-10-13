import pandas as pd
import numpy as np
import pickle
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2
import rpy2.robjects as ro
import argparse
import os
import logging
import json
from sklearn.metrics import roc_auc_score, accuracy_score

# Convert pandas.DataFrames to R dataframes automatically.
pandas2ri.activate()


def load_data(data_name):
    def transform(df):
        df = df.drop_duplicates()
        # df['score'] = df['score'].apply(lambda x: -1 if x == 0 else 1)
        return df

    train_data = pd.read_csv("data/{}/train.csv".format(data_name))
    valid_data = pd.read_csv("data/{}/valid.csv".format(data_name))
    test_data = pd.read_csv("data/{}/test.csv".format(data_name))
    train_set, valid_set, test_set = [transform(data_set) for data_set in [train_data, valid_data, test_data]]

    user_n = np.max(train_data['user_id'])
    item_n = np.max([np.max(train_data['item_id']), np.max(valid_data['item_id']), np.max(test_data['item_id'])])

    ro.r.assign("train_set", train_set)
    # user spread() in R language to change the dataframe into response matrix
    ro.r("""
    library(tidyr)
    train_set <- spread(train_set, item_id, score)
    rownames(train_set) <- train_set$user_id
    train_set <- train_set[,-1]
    """)  # user_id must be the first column
    return user_n, item_n, ro.r['train_set'], valid_set, test_set


def gpirt_train(data, sample_iterations=1000, burn_iterations=500):
    '''
    :param data:
    :param sample_iterations:
    :param burn_iterations:
    The gpirt (R package) will delete unanimous questions before training
    :return:
    '''
    print('training...')
    logging.info('training...')
    gpirt = importr('gpirt')
    samples = gpirt.gpirtMCMC(data, sample_iterations, burn_iterations, vote_codes=ro.r(r"list(yea=1, nay=0, missing=c(NA,NaN))"))
    logging.info('finished')
    results = {}
    for i, name in enumerate(samples.names):  # ['theta', 'beta', 'f', 'IRFs']
        results[name] = samples[i]
    return results


def get_unanimous_projection(data):
    '''
    :param data:
    :return:
    '''
    remained_items = []
    for col_name in data.columns:
        tmp = data[col_name]
        tmp = tmp[tmp >= -1].unique()
        if len(tmp) > 1:
            remained_items.append(col_name)  #
    remained_itemIdx2idx = {item: i + 1 for i, item in enumerate(remained_items)}  # idx starts from 1
    return remained_itemIdx2idx


def get_predict(thetas, itemIds, gpirt_params, remained_itemId2idx):
    thetas = np.clip(thetas, -5, 5)   # gpirt only returns the IRFs values in [-5, 5]
    thetas = (np.round(thetas * 100) + 500).astype(int)
    irfs = gpirt_params['IRFs']

    preds = []
    for i, theta in enumerate(thetas):
        item_id = str(itemIds[i])
        if item_id in remained_itemId2idx:
            idx = remained_itemId2idx[item_id]
            preds.append(irfs[theta][idx - 1])
        else:
            preds.append(np.NaN)
    return preds


def test(data_set, gpirt_params, remained_itemId2idx):
    logging.info('test ...')
    user_ids = data_set['user_id']   # start from 1
    item_ids = data_set['item_id']   # start from 1
    theta_means_all = gpirt_params['theta'].mean(axis=0)
    theta_means = theta_means_all[user_ids - 1]
    preds = get_predict(theta_means, item_ids, gpirt_params, remained_itemId2idx)

    preds = np.array(preds)
    ys = data_set['score']
    # handle nan: 1. the questions in test set that didn't exist in training set; 2. unanimous questions
    mask = ~np.isnan(preds)
    preds, ys = preds[mask], ys[mask]

    auc = roc_auc_score(ys, preds)
    acc = accuracy_score(ys, np.array(preds) >= 0.5)
    logging.info(f'auc={auc}, acc={acc}')
    print(f'auc={auc}, acc={acc}')


# ========================= interval related metrics ========================-
def calculate_interval_metric(data_set, gpirt_params, remained_itemId2idx, d, mu, eta, percentage):
    logging.info('calculate_interval_metric ... d={}, mu={}, eta={}'.format(d, mu, eta))
    ys = data_set['score'].values
    user_ids = data_set['user_id']   # start from 1
    item_ids = data_set['item_id']   # start from 1
    theta_means_all = gpirt_params['theta'].mean(axis=0)
    theta_stds_all = gpirt_params['theta'].std(axis=0)

    theta_up = theta_means_all[user_ids - 1] + theta_stds_all[user_ids - 1] * d
    theta_down = theta_means_all[user_ids - 1] - theta_stds_all[user_ids - 1] * d
    preds_up = get_predict(theta_up, item_ids, gpirt_params, remained_itemId2idx)
    preds_down = get_predict(theta_down, item_ids, gpirt_params, remained_itemId2idx)

    mask = ~np.isnan(preds_up)
    preds_up, preds_down, ys = np.array(preds_up)[mask], np.array(preds_down)[mask], ys[mask]

    preds_up = np.array(preds_up).reshape(1, -1)
    preds_down = np.array(preds_down).reshape(1, -1)
    picp = PICP_pred(preds_up, preds_down, ys, percentage) / len(ys)
    pinaw = PINAW(preds_up, preds_down, reduce=True)
    cwc = CWC(picp, pinaw, mu, eta)
    print('PICP={}, PINAW={}, CWC={}'.format(picp, pinaw, cwc))
    logging.info('PICP={}, PINAW={}, CWC={}'.format(picp, pinaw, cwc))


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
    '''

    :param picp:
    :param pinaw:
    :param mu:
    :param eta:
    :return:
    '''
    gamma = 1 if picp < mu else 0
    return pinaw * (1 + gamma * np.exp(eta * (mu - picp)))


def tmp(data_name):
    df = pd.read_csv(f'data/{data_name}/train.csv')
    df.sort_values(by=['user_id', 'item_id'], inplace=True)
    df.to_csv(f'data/{data_name}/train.csv', index=False)


if __name__ == '__main__':
    model_name = 'GPIRT'

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='data name')
    args = parser.parse_args()

    data_name = args.data
    dst_folder = 'result/{}/{}'.format(data_name, model_name)
    if not os.path.isdir(dst_folder):
        os.makedirs(dst_folder)
    logging.basicConfig(
        filename='{}/{}.log'.format(dst_folder, model_name),
        level=logging.INFO, format='%(asctime)s %(message)s')

    user_n, item_n, train_data, valid_data, test_data = load_data(data_name)
    # # crmsg = CaptureRMessage()
    # # crmsg.capture_r_msg()

    # train and save trained parameters
    params = gpirt_train(train_data, 10, 5)
    with open(f'{dst_folder}/params.dict.pkl', 'wb') as f:
        pickle.dump(params, f)
    itemId2idx = get_unanimous_projection(train_data)
    with open(f'{dst_folder}/remained_itemIdx2idx.json', 'w') as f:
        json.dump(itemId2idx, f)

    # load parameters
    # with open(f'{dst_folder}/params.dict.pkl', 'rb') as f:
    #     params = pickle.load(f)
    # with open(f'{dst_folder}/remained_itemIdx2idx.json', 'r') as f:
    #     itemId2idx = json.load(f)

    # test
    test(test_data, params, itemId2idx)
    calculate_interval_metric(test_data, params, itemId2idx, d=1.96, mu=0.95, eta=10, percentage=False)



