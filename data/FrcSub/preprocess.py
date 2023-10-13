import numpy as np
import pandas as pd
import csv
from random import shuffle


def transform_data():
    '''
    Transform the format of data into data_all.csv
    then divide the data into train set, valid set and test set
    :return:
    '''
    data_np = np.loadtxt('data.txt')
    stu_n, exer_n = data_np.shape

    # all data
    with open('data_all.csv', 'w', encoding='utf8') as o_f:
        o_f.write('user_id,item_id,score\n')
        for stu_i in range(stu_n):
            for exer_j in range(exer_n):
                o_f.write('{},{},{}\n'.format(stu_i + 1, exer_j + 1, int(data_np[stu_i][exer_j])))  # idx starts from 1

    # split data
    idxes = list(range(exer_n))
    train_size, val_size = int(exer_n * 0.7), int(exer_n * 0.1)
    test_size = exer_n - train_size - val_size
    train_set, val_set, test_set = [], [], []
    for stu_i in range(stu_n):
        shuffle(idxes)
        train_set.extend([[stu_i + 1, idxes[j] + 1, data_np[stu_i, idxes[j]]] for j in range(train_size)])
        val_set.extend([[stu_i + 1, idxes[j] + 1, data_np[stu_i, idxes[j]]] for j in range(train_size, train_size + val_size)])
        test_set.extend(([[stu_i + 1, idxes[j] + 1, data_np[stu_i, idxes[j]]] for j in range(- test_size, 0)]))
    with open('train.csv', 'w', encoding='utf8') as o_f:
        csv_writer = csv.writer(o_f)
        csv_writer.writerow(['user_id', 'item_id', 'score'])
        csv_writer.writerows(train_set)
    with open('valid.csv', 'w', encoding='utf8') as o_f:
        csv_writer = csv.writer(o_f)
        csv_writer.writerow(['user_id', 'item_id', 'score'])
        csv_writer.writerows(val_set)
    with open('test.csv', 'w', encoding='utf8') as o_f:
        csv_writer = csv.writer(o_f)
        csv_writer.writerow(['user_id', 'item_id', 'score'])
        csv_writer.writerows(test_set)


def transform_qmatrix():
    qmatrix = np.loadtxt('q.txt')
    exer_n, knowledge_n = qmatrix.shape
    kn_idx = np.arange(1, knowledge_n + 1)
    items = []
    for i in range(exer_n):
        items.append([i + 1, list(kn_idx[qmatrix[i] == 1])])
    with open('item.csv', 'w', encoding='utf8') as o_f:
        csv_writer = csv.writer(o_f)
        csv_writer.writerow(['item_id', 'knowledge_code'])
        csv_writer.writerows(items)


transform_data()
transform_qmatrix()

