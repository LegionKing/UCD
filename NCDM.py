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

batch_size = 32


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * torch.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class Net(nn.Module):

    def __init__(self, exer_n, student_n, knowledge_n, stu_info_summary, exer_info_summary, kn_info_summary):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        # self.stu_n = student_n
        self.emb_num = student_n
        self.stu_info_summary = stu_info_summary
        self.exer_info_summary = exer_info_summary
        self.kn_info_summary = kn_info_summary
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        self.prednet = nn.Sequential(
            PosLinear(self.prednet_input_len, self.prednet_len1),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            PosLinear(self.prednet_len1, self.prednet_len2),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            PosLinear(self.prednet_len2, 1),
            nn.Sigmoid()
        )

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, exer_knowledge_point, y, stage, device):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_difficulty(exer_id))

        # prednet
        input_x = e_discrimination * (stat_emb - k_difficulty) * exer_knowledge_point
        output_1 = self.prednet(input_x)

        return output_1.view(-1)

    def get_student_states(self, stu_id):
        if stu_id is None:
            stu_id = torch.arange(0, len(self.stu_info_summary)).long().to(self.student_emb.weight.device)
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        return stat_emb.detach()


class NCDM(CDM):
    def __init__(self, **kwargs):
        super(NCDM, self).__init__()
        self.net = Net(kwargs['exer_n'], kwargs['student_n'], kwargs['knowledge_n'], kwargs['stu_info_summary'],
                       kwargs['exer_info_summary'], kwargs['kn_info_summary'])

    def train(self, train_set, valid_set, test_set, data_name, dst_folder, model_name='NCDM', stage=1, lr=0.02, device='cpu', epoch_n=15):
        logging.info("training... (lr={})".format(lr))
        self.net = self.net.to(device)
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        for epoch_i in range(epoch_n):
            self.net.train()
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_set, "Epoch %s, Stage %d" % (epoch_i, stage)):
                batch_count += 1
                user_info, item_info, knowledge_emb, y = batch_data
                user_info: torch.Tensor = user_info.to(device)
                item_info: torch.Tensor = item_info.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = y.to(device)
                pred = self.net(user_info, item_info, knowledge_emb, y, stage=stage, device=device)
                loss = loss_function(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            print("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))
            logging.info("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))
            auc, acc = self.eval(valid_set, device, stage)
            print("[Epoch %d] | Valid set, auc: %.6f, acc: %.6f" % (epoch_i, auc, acc))
            logging.info("[Epoch %d] | Valid set, auc: %.6f, acc: %.6f" % (epoch_i, auc, acc))
            auc, acc = self.eval(test_set, device, stage)
            print("[Epoch %d] | Test set, auc: %.6f, acc: %.6f" % (epoch_i, auc, acc))
            logging.info("[Epoch %d] | Test set, auc: %.6f, acc: %.6f" % (epoch_i, auc, acc))

            if not os.path.isdir(dst_folder):
                os.makedirs(dst_folder)
            self.save(os.path.join(dst_folder, '{}-lr{}.snapshot{}'.format(model_name, lr, epoch_i)))
        return auc, acc

    def eval(self, test_data, device="cpu", stage=1):
        logging.info('eval ... ')
        self.net = self.net.to(device)
        self.net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred = self.net(user_id, item_id, knowledge_emb, y, stage=stage, device=device)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def eval_prob(self, test_data, device="cpu", stage=1):
        '''
        only output the predicted probabilities
        :return:
        '''
        logging.info('eval_prob ... ')
        self.net = self.net.to(device)
        self.net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred = self.net(user_id, item_id, knowledge_emb, y, stage=stage, device=device)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        return y_true, y_pred

    def get_pars(self):
        stat = torch.sigmoid(self.net.student_emb.weight.data)
        diff = torch.sigmoid(self.net.k_difficulty.weight.data)
        disc = torch.sigmoid(self.net.e_difficulty.weight.data)
        return stat, diff, disc

    def get_stu_pars(self):
        stat = torch.sigmoid(self.net.student_emb.weight.data.cpu())
        return stat.numpy()

    def save(self, filepath):
        torch.save(self.net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.net.load_state_dict(torch.load(filepath, map_location=lambda s, loc: s))
        logging.info("load parameters from %s" % filepath)


def transform(user, item, item2knowledge, score, batch_size, knowledge_n, shuffle=False):
    knowledge_emb = torch.zeros((len(item), knowledge_n))
    for idx in range(len(item)):
        knowledge_emb[idx][np.array(item2knowledge[item[idx]]) - 1] = 1.0

    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64) - 1,  # (1, user_n) to (0, user_n-1)
        torch.tensor(item, dtype=torch.int64) - 1,  # (1, item_n) to (0, item_n-1)
        knowledge_emb,
        torch.tensor(score, dtype=torch.float32)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)


def load_data(data_name):
    train_data = pd.read_csv("data/{}/train.csv".format(data_name))
    valid_data = pd.read_csv("data/{}/valid.csv".format(data_name))
    test_data = pd.read_csv("data/{}/test.csv".format(data_name))
    df_item = pd.read_csv("data/{}/item.csv".format(data_name))
    item2knowledge = {}
    knowledge_set = set()
    for i, s in df_item.iterrows():
        item_id, knowledge_codes = s['item_id'], list(set(eval(s['knowledge_code'])))
        item2knowledge[item_id] = knowledge_codes
        knowledge_set.update(knowledge_codes)

    user_n = np.max(train_data['user_id'])
    item_n = np.max([np.max(train_data['item_id']), np.max(valid_data['item_id']), np.max(test_data['item_id'])])
    knowledge_n = np.max(list(knowledge_set))

    train_set = transform(train_data["user_id"], train_data["item_id"], item2knowledge, train_data["score"], batch_size, knowledge_n, shuffle=True)
    valid_set, test_set = [
        transform(data["user_id"], data["item_id"], item2knowledge, data["score"], batch_size, knowledge_n)
        for data in [valid_data, test_data]
    ]

    return user_n, item_n, knowledge_n, train_set, valid_set, test_set


if __name__ == '__main__':
    model_name = 'NCDM'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='FrcSub', help='data name')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.002)
    args = parser.parse_args()

    data_name = args.data
    dst_folder = 'result/{}/{}'.format(data_name, model_name)
    if not os.path.isdir(dst_folder):
        os.makedirs(dst_folder)
    logging.basicConfig(
        filename='{}/{}-lr{}.log'.format(dst_folder, model_name, args.lr),
        level=logging.INFO, format='%(asctime)s %(message)s')

    stu_n, exer_n, knowledge_n, train_set, valid_set, test_set = load_data(data_name)
    ncdm = NCDM(exer_n=exer_n, student_n=stu_n, knowledge_n=knowledge_n, stu_info_summary=None, exer_info_summary=None, kn_info_summary=None)
    ncdm.train(train_set, valid_set, test_set, data_name, dst_folder, model_name=model_name, device=args.device, epoch_n=args.epoch, lr=args.lr)
