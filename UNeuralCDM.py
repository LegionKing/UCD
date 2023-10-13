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
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman', size=20)
import random

batch_size = 32


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor, sample: bool) -> torch.Tensor:
        weight = torch.abs(self.weight)
        return F.linear(input, weight, self.bias), 0, 0


class BayesPosLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, prior: Distribution, bias: bool = True):
        super(BayesPosLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior = prior
        self.w_mu = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.w_std_eta = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        if bias:
            self.b_mu = nn.Parameter(torch.Tensor(self.out_features))
            self.b_std_eta = nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('b_mu', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        with torch.no_grad():
            self.w_mu.set_(torch.log(torch.abs(self.w_mu.data)))
        nn.init.uniform_(self.w_std_eta, -4., -2.)
        if self.b_mu is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_mu)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.b_mu, -bound, bound)
            nn.init.uniform_(self.b_std_eta, -bound, bound)

    def forward(self, input: torch.Tensor, sample: bool = True):
        if not self.training or not sample:
            w = torch.exp(self.w_mu)    # median
            output = F.linear(input, w, self.b_mu)
            return output, 0, 0
        else:
            sample_n, batch_size, in_feature = input.size()
            eps_w = self.w_mu.data.new(batch_size, self.w_mu.size()[0], self.w_mu.size()[1]).normal_()
            eps_b = self.b_mu.data.new(batch_size, self.b_mu.size()[0]).normal_()
            std_w = 1e-6 + F.softplus(self.w_std_eta)
            std_b = 1e-6 + F.softplus(self.b_std_eta)
            w = torch.exp(self.w_mu + std_w * eps_w)
            b = self.b_mu + std_b * eps_b
            output_t = input.data.new(batch_size, sample_n, self.out_features)
            input_t = torch.transpose(input, 0, 1)
            for i in range(batch_size):
                output_t[i] = F.linear(input_t[i], w[i], b[i])
            output = torch.transpose(output_t, 0, 1)

            kl_w = kl.kl_divergence(normal.Normal(self.w_mu, std_w), self.prior)
            kl_b = kl.kl_divergence(normal.Normal(self.b_mu, std_b), self.prior)
            return output, kl_w.sum(), kl_b.sum()


class Net(nn.Module):

    def __init__(self, exer_n, student_n, knowledge_n, stu_info_summary, exer_info_summary, kn_info_summary):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.stu_n = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256

        super(Net, self).__init__()

        # prediction sub-net
        self.lambda_1_eta_stu, self.lambda_2_eta_stu = nn.Parameter(torch.Tensor([1.])), nn.Parameter(torch.Tensor([1.]))
        self.lambda_1_eta_exer, self.lambda_2_eta_exer = nn.Parameter(torch.Tensor([1.])), nn.Parameter(torch.Tensor([1.]))
        self.stu_cnt: torch.Tensor
        self.exer_cnt: torch.Tensor
        self.info_prior(stu_info_summary, exer_info_summary, kn_info_summary)

        self.student_mean = nn.Embedding(self.stu_n, self.stu_dim)
        self.student_std_eta = nn.Embedding(self.stu_n, self.stu_dim)
        self.k_diff_mean = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.k_diff_std_eta = nn.Embedding(self.exer_n, self.knowledge_dim)   # std = softplus(eta)
        self.e_disc_mean = nn.Embedding(self.exer_n, 1)
        self.e_disc_std_eta = nn.Embedding(self.exer_n, 1)
        self.prior = normal.Normal(0, 1)

        self.bpl_1 = BayesPosLinear(self.prednet_input_len, self.prednet_len1, prior=self.prior)
        # self.bpl_1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.bpl_2 = BayesPosLinear(self.prednet_len1, self.prednet_len2, prior=self.prior)
        # self.bpl_2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.bpl_3 = BayesPosLinear(self.prednet_len2, 1, prior=self.prior)
        # self.bpl_3 = PosLinear(self.prednet_len2, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def get_diag_distribution(self, stu_id, exer_id):
        stat_mean = self.student_mean(stu_id)
        stat_std_model = F.softplus(self.student_std_eta(stu_id))
        stat_std_data = F.softplus(self.lambda_1_eta_stu) * torch.exp(- F.softplus(self.lambda_2_eta_stu) * self.stu_cnt[stu_id])
        stat_std = combine_std_(stat_std_data, stat_std_model)

        k_diff_mean = self.k_diff_mean(exer_id)
        k_diff_std_model = F.softplus(self.k_diff_std_eta(exer_id))
        exer_std_data = F.softplus(self.lambda_1_eta_exer) * torch.exp(- F.softplus(self.lambda_2_eta_exer) * self.exer_cnt[exer_id])
        k_diff_std = combine_std_(exer_std_data, k_diff_std_model)

        e_disc_mean = self.e_disc_mean(exer_id)
        e_disc_std_model = F.softplus(self.e_disc_std_eta(exer_id))
        e_disc_std = combine_std_(exer_std_data, e_disc_std_model)

        return stat_mean, stat_std, k_diff_mean, k_diff_std, e_disc_mean, e_disc_std

    def get_kl(self, mean, std, prior):
        return kl.kl_divergence(normal.Normal(mean, std + 1e-5), prior).sum()

    def forward(self, stu_id, exer_id, exer_knowledge_point, y, device, sample=True, sample_n=1):
        # before prednet
        stat_mean, stat_std, k_diff_mean, k_diff_std, e_disc_mean, e_disc_std = self.get_diag_distribution(stu_id, exer_id)

        if sample:
            stu_stat = torch.sigmoid(self.reparameterize_gaussian(stat_mean, stat_std, sample_n))  # (sample_n, batch_size, knowledge_n)
            k_difficulty = torch.sigmoid(self.reparameterize_gaussian(k_diff_mean, k_diff_std, sample_n))  # (sample_n, batch_size, knowledge_n)
            e_discrimination = torch.sigmoid(self.reparameterize_gaussian(e_disc_mean, e_disc_std, sample_n))  # (sample_n, batch_size, knowledge_n)
        else:
            stu_stat = torch.sigmoid(stat_mean)
            k_difficulty = torch.sigmoid(k_diff_mean)
            e_discrimination = torch.sigmoid(e_disc_mean)

        # prednet
        kl_w_sum, kl_b_sum = 0, 0
        input_x = e_discrimination * (stu_stat - k_difficulty) * exer_knowledge_point
        input_x, kl_w, kl_b = self.bpl_1(input_x, sample=sample)
        kl_w_sum, kl_b_sum = kl_w_sum + kl_w, kl_b_sum + kl_b
        input_x = torch.sigmoid(input_x)
        # input_x = self.drop_1(input_x)
        input_x, kl_w, kl_b = self.bpl_2(input_x, sample=sample)
        kl_w_sum, kl_b_sum = kl_w_sum + kl_w, kl_b_sum + kl_b
        input_x = torch.sigmoid(input_x)
        # input_x = self.drop_2(input_x)
        output_1, kl_w, kl_b = self.bpl_3(input_x, sample=sample)
        kl_w_sum, kl_b_sum = kl_w_sum + kl_w, kl_b_sum + kl_b
        output_1 = torch.sigmoid(output_1)

        # get diag_kl divergence
        stu_id, exer_id = torch.arange(self.stu_n).long().to(device), torch.arange(self.exer_n).long().to(device)
        all_stat_mean, all_stat_std, all_k_diff_mean, all_k_diff_std, all_e_disc_mean, all_e_disc_std = self.get_diag_distribution(stu_id, exer_id)
        kl_stat = self.get_kl(all_stat_mean, all_stat_std, self.prior)
        kl_diff = self.get_kl(all_k_diff_mean, all_k_diff_std, self.prior)
        kl_disc = self.get_kl(all_e_disc_mean, all_e_disc_std, self.prior)
        kl_diag = kl_stat + kl_diff + kl_disc

        return output_1.view(sample_n, -1) if sample else output_1.view(-1), kl_w_sum, kl_b_sum, kl_diag

    def pred_interval(self, stu_id, exer_id, exer_knowledge_point, d, device, sample_n=10):
        # before prednet
        stat_mean = self.student_mean(stu_id)
        stat_std = F.softplus(self.student_std_eta(stu_id))
        stat_up = torch.sigmoid(stat_mean + d * stat_std)
        stat_down = torch.sigmoid(stat_mean - d * stat_std)

        k_diff_mean = self.k_diff_mean(exer_id)
        k_diff_std = F.softplus(self.k_diff_std_eta(exer_id))
        k_difficulty = torch.sigmoid(self.reparameterize_gaussian(k_diff_mean, k_diff_std, sample_n))  # (sample_n, batch_size, knowledge_n)

        e_disc_mean = self.e_disc_mean(exer_id)
        e_disc_std = F.softplus(self.e_disc_std_eta(exer_id))
        e_discrimination = torch.sigmoid(self.reparameterize_gaussian(e_disc_mean, e_disc_std, sample_n)) # (sample_n, batch_size, knowledge_n)

        # prednet
        input_x = e_discrimination * (stat_up - k_difficulty) * exer_knowledge_point
        input_x, _, _ = self.bpl_1(input_x, sample=False)
        input_x = torch.sigmoid(input_x)
        # input_x = self.drop_1(input_x)
        input_x, _, _ = self.bpl_2(input_x, sample=False)
        input_x = torch.sigmoid(input_x)
        # input_x = self.drop_1(input_x)
        input_x, _, _ = self.bpl_3(input_x, sample=False)
        output_1_up = torch.sigmoid(input_x)

        input_x = e_discrimination * (stat_down - k_difficulty) * exer_knowledge_point
        input_x, _, _ = self.bpl_1(input_x, sample=False)
        input_x = torch.sigmoid(input_x)
        # input_x = self.drop_1(input_x)
        input_x, _, _ = self.bpl_2(input_x, sample=False)
        input_x = torch.sigmoid(input_x)
        # input_x = self.drop_1(input_x)
        input_x, _, _ = self.bpl_3(input_x, sample=False)
        output_1_down = torch.sigmoid(input_x)

        return output_1_up.view(sample_n, -1), output_1_down.view(sample_n, -1)

    def info_prior(self, stu_info_summary, exer_info_summary, kn_info_summary):
        self.stu_cnt = stu_info_summary[:, :self.knowledge_dim] + stu_info_summary[:, self.knowledge_dim:]
        diff_cnt = exer_info_summary[:, :self.knowledge_dim] + exer_info_summary[:, self.knowledge_dim:]
        self.exer_cnt = torch.zeros(diff_cnt.shape[0], 1)  # (exer_n, 1)
        for i in range(diff_cnt.shape[0]):
            a = (diff_cnt[i] > 0).int().sum()  # the number of knowledge concepts of this question
            if a > 0:
                self.exer_cnt[i][0] = diff_cnt[i].sum() / a

    @staticmethod
    def reparameterize_gaussian(mean_, std_, sample_n):
        batch_shape = mean_.size()
        eps = _standard_normal(torch.Size((sample_n, batch_shape[0], batch_shape[1])), dtype=mean_.dtype, device=mean_.device)
        return mean_ + std_ * eps   # broadcast


class UNCDM(CDM):
    def __init__(self, **kwargs):
        super(UNCDM, self).__init__()
        self.net = Net(kwargs['exer_n'], kwargs['student_n'], kwargs['knowledge_n'], kwargs['stu_info_summary'],
                       kwargs['exer_info_summary'], kwargs['kn_info_summary'])

    def train(self, train_set, valid_set, test_set, stage=1, lr=0.002, device='cpu', epoch_n=15, sample_n=1, net_klw=0., diag_klw=0.):
        '''
        :param train_set:
        :param valid_set:
        :param test_set:
        :param stage:
        :param lr:
        :param device:
        :param epoch_n:
        :return:
        '''
        logging.info("training... (lr={}, sample_n={}, net_klw={}, diag_klw={}, combine={}".format(lr, sample_n, net_klw, diag_klw, combine_method))
        self.net = self.net.to(device)
        self.net.stu_cnt = self.net.stu_cnt.to(device)
        self.net.exer_cnt = self.net.exer_cnt.to(device)
        loss_function = nn.BCELoss(reduction='none')
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        batch_n = len(train_set)
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
                pred, kl_w_sum, kl_b_sum, kl_diag_sum = self.net(user_info, item_info, knowledge_emb, y, device=device, sample_n=sample_n)
                recovery_loss = loss_function(pred, y.view(1, -1).repeat(sample_n, 1)).mean(dim=0).sum()
                pi_i = 2 ** (batch_n - batch_count) / (2 ** batch_n - 1)
                loss = recovery_loss + net_klw * (kl_w_sum + kl_b_sum) * pi_i + diag_klw * kl_diag_sum / batch_n
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            logging.info("[Epoch %d] average training loss of batch: %.6f" % (epoch_i, float(np.mean(epoch_losses))))
            print("[Epoch %d] average training loss of batch: %.6f" % (epoch_i, float(np.mean(epoch_losses))))
            auc, acc = self.eval(valid_set, device, 10)
            logging.info("[Epoch %d] | valid set, auc: %.6f, acc: %.6f" % (epoch_i, auc, acc))
            print("[Epoch %d] | valid set, auc: %.6f, acc: %.6f" % (epoch_i, auc, acc))
            auc, acc = self.eval(test_set, device, 10)
            logging.info("[Epoch %d] | test set, auc: %.6f, acc: %.6f" % (epoch_i, auc, acc))
            print("[Epoch %d] | test set, auc: %.6f, acc: %.6f" % (epoch_i, auc, acc))

            dst_folder = 'result/{}/{}'.format(data_name, model_name)
            if not os.path.isdir(dst_folder):
                os.makedirs(dst_folder)
            global model_prefix
            self.save('{}{}'.format(model_prefix, epoch_i))

        return auc, acc

    def eval(self, test_data, device="cpu", stage=1, sample_n=1):
        logging.info('eval ... (sample_n={})'.format(sample_n))
        self.net = self.net.to(device)
        self.net.stu_cnt = self.net.stu_cnt.to(device)
        self.net.exer_cnt = self.net.exer_cnt.to(device)
        self.net.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch_data in tqdm(test_data, "Evaluating"):
                user_id, item_id, knowledge_emb, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                pred, _, _, _ = self.net(user_id, item_id, knowledge_emb, y, device=device, sample=True, sample_n=sample_n)
                pred = pred.mean(dim=0)
                y_pred.extend(pred.detach().cpu().tolist())
                y_true.extend(y.tolist())

        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def eval_cwc(self, test_data, d, mu, eta, device="cpu", sample_n=20):
        logging.info('eval_cwc ... d={}, mu={}, eta={}, sample_n={}'.format(d, mu, eta, sample_n))
        self.net = self.net.to(device)
        self.net.stu_cnt = self.net.stu_cnt.to(device)
        self.net.exer_cnt = self.net.exer_cnt.to(device)
        self.net.eval()
        cover_cnt, pinaw_sum, data_cnt = 0, 0, 0
        with torch.no_grad():
            for batch_data in tqdm(test_data, "Evaluating"):
                user_id, item_id, knowledge_emb, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = y.to(device)
                up_exp, down_exp = self.net.pred_interval(user_id, item_id, knowledge_emb, d, device=device, sample_n=sample_n)
                # up_exp, down_exp = self.net.pred_interval_mu(user_id, item_id, knowledge_emb, d, device=device)
                cover_cnt += PICP_pred(up_exp, down_exp, y)
                pinaw_sum += PINAW(up_exp, down_exp).sum()
                data_cnt += len(y)
        picp = cover_cnt / data_cnt
        pinaw = pinaw_sum / data_cnt
        cwc = CWC(picp, pinaw, mu=mu, eta=eta)
        return picp, pinaw, cwc

    def analysis_std_model1(self, data_set, device="cpu"):
        self.net = self.net.to(device)
        self.net.stu_cnt = self.net.stu_cnt.to(device)
        self.net.exer_cnt = self.net.exer_cnt.to(device)
        self.net.eval()
        stu_std_model = F.softplus(self.net.student_std_eta.weight.data)
        pred_differ = torch.zeros(self.net.stu_n, self.net.knowledge_dim)
        pred_cnt = torch.zeros(self.net.stu_n, self.net.knowledge_dim)
        with torch.no_grad():
            for batch_data in data_set:
                user_id_cpu, item_id_cpu, knowledge_emb_cpu, y_cpu = batch_data
                user_id: torch.Tensor = user_id_cpu.to(device)
                item_id: torch.Tensor = item_id_cpu.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb_cpu.to(device)
                pred, _, _, _ = self.net(user_id, item_id, knowledge_emb, y_cpu, device=device, sample=False)
                difference = torch.abs(pred.cpu() - y_cpu)
                for i in range(len(user_id_cpu)):
                    mask = knowledge_emb_cpu[i] == 1
                    pred_differ[user_id_cpu[i]][mask] += difference[i]
                    pred_cnt[user_id_cpu[i]][mask] += 1
        stu_std_model = pd.Series(stu_std_model.view(-1).cpu().numpy())
        pred_differ_sum = pd.Series(pred_differ.view(-1).numpy())
        pred_differ_avg = pd.Series((pred_differ / pred_cnt).view(-1).numpy())
        assert stu_std_model.size == pred_differ_sum.size
        corr_sum = stu_std_model.corr(pred_differ_sum, method='spearman')
        corr_avg = stu_std_model.corr(pred_differ_avg, method='spearman')
        return corr_sum, corr_avg, stu_std_model.mean(), pred_differ_sum.mean(), pred_differ_avg.mean()

    def get_pars(self):
        self.net = self.net.cpu()

        with torch.no_grad():
            stu_mean = self.net.student_mean.weight.data
            stu_std_model = F.softplus(self.net.student_std_eta.weight.data)
            stu_std_data = F.softplus(self.net.lambda_1_eta_stu) * torch.exp(
                - F.softplus(self.net.lambda_2_eta_stu) * self.net.stu_cnt.cpu())
            k_diff_mean = self.net.k_diff_mean.weight.data
            k_diff_std_model = F.softplus(self.net.k_diff_std_eta.weight.data)
            e_std_data = F.softplus(self.net.lambda_1_eta_exer) * torch.exp(
                - F.softplus(self.net.lambda_2_eta_exer) * self.net.exer_cnt.cpu())
            e_disc_mean = self.net.e_disc_mean.weight.data
            e_disc_std_model = F.softplus(self.net.e_disc_std_eta.weight.data)
        return stu_mean, stu_std_model, stu_std_data, k_diff_mean, k_diff_std_model, e_disc_mean, e_disc_std_model, e_std_data

    def get_stu_pars(self):
        self.net = self.net.cpu()
        self.net.stu_cnt = self.net.stu_cnt.cpu()
        self.net.exer_cnt = self.net.exer_cnt.cpu()
        stu_id = torch.arange(self.net.stu_cnt.size()[0])
        with torch.no_grad():
            stat_mean, stat_std, _, _, _, _ = self.net.get_diag_distribution(stu_id, torch.tensor([0]))
        return [stat_mean.numpy(), stat_std.numpy()]

    def save(self, filepath):
        torch.save(self.net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.net.load_state_dict(torch.load(filepath, map_location=lambda s, loc: s))
        logging.info("load parameters from %s" % filepath)


def combine_std_(std_data, std_model):
    global combine_method
    if combine_method == '+':
        std = std_data + std_model
    elif combine_method == 'x':
        std = std_data * std_model
    else:
        print('unrecognized combing_method')
        exit(1)
    return std


def transform(user, item, item2knowledge, score, batch_size, knowledge_n):
    knowledge_emb = torch.zeros((len(item), knowledge_n))
    for idx in range(len(item)):
        knowledge_emb[idx][np.array(item2knowledge[item[idx]]) - 1] = 1.0

    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64) - 1,  # (1, user_n) to (0, user_n-1)
        torch.tensor(item, dtype=torch.int64) - 1,  # (1, item_n) to (0, item_n-1)
        knowledge_emb,
        torch.tensor(score, dtype=torch.float32)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


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

    train_set, valid_set, test_set = [
        transform(data["user_id"], data["item_id"], item2knowledge, data["score"], batch_size, knowledge_n)
        for data in [train_data, valid_data, test_data]
    ]

    return user_n, item_n, knowledge_n, train_set, valid_set, test_set


def load_data_info(data_name):
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

    # format train set
    info_summary_fname = "data/{}/info_summary.pkl".format(data_name)
    if os.path.exists(info_summary_fname):
        with open(info_summary_fname, 'rb') as i_f:
            stu_info_summary, exer_info_summary, kn_info_summary = pickle.load(i_f)
    else:
        stu_info_summary = torch.zeros((user_n, 2 * knowledge_n))
        exer_info_summary = torch.zeros((item_n, 2 * knowledge_n))
        kn_info_summary = torch.zeros(2 * knowledge_n)
        knowledge_emb = torch.zeros((len(train_data), knowledge_n))
        for i, s in train_data.iterrows():
            knowledge_emb[i][np.array(item2knowledge[s['item_id']]) - 1] = 1.0
            stu_idx, exer_idx, score = int(s['user_id']) - 1, int(s['item_id']) - 1, s['score']
            knowledge = knowledge_emb[i]
            if score == 1:
                stu_info_summary[stu_idx][:knowledge_n] = stu_info_summary[stu_idx][:knowledge_n] + knowledge
                exer_info_summary[exer_idx][:knowledge_n] = exer_info_summary[exer_idx][:knowledge_n] + knowledge
                kn_info_summary[:knowledge_n] = kn_info_summary[:knowledge_n] + knowledge
            else:
                stu_info_summary[stu_idx][knowledge_n:] = stu_info_summary[stu_idx][knowledge_n:] + knowledge
                exer_info_summary[exer_idx][knowledge_n:] = exer_info_summary[exer_idx][knowledge_n:] + knowledge
                kn_info_summary[knowledge_n:] = kn_info_summary[knowledge_n:] + knowledge
        with open(info_summary_fname, 'wb') as o_f:
            pickle.dump((stu_info_summary, exer_info_summary, kn_info_summary), o_f)

    train_set, valid_set, test_set = [
        transform(data["user_id"], data["item_id"], item2knowledge, data["score"], batch_size, knowledge_n)
        for data in [train_data, valid_data, test_data]
    ]

    return user_n, item_n, knowledge_n, train_set, valid_set, test_set, stu_info_summary, exer_info_summary, kn_info_summary


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
        l = torch.abs(y - pred_down) < 0.5
        r = torch.abs(y - pred_up) < 0.5
        cover_cnt = (l | r).float().mean(dim=0).sum()
    else:
        percent_1 = torch.zeros(sample_n, batch_size).to(pred_up.device)
        for i in range(sample_n):
            for j in range(batch_size):
                if pred_down[i][j] >= 0.5:
                    percent_1[i][j] = 1
                elif pred_up[i][j] > 0.5:
                    percent_1[i][j] = (pred_up[i][j] - 0.5) / (pred_up[i][j] - pred_down[i][j])
        cover_cnt = (percent_1 * y + (1 - y) * (1 - percent_1)).mean(dim=0).sum()
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
    a = (up - down).mean(dim=0) / range_y
    if reduce:
        a = a.mean()
    return a


def CWC(picp, pinaw, mu=0.95, eta=10):
    gamma = 1 if picp < mu else 0
    return pinaw * (1 + gamma * torch.exp(eta * (mu - picp)))


def calculate_interval_metric(data_set, model: UNCDM, path_prefix, epochs, d, mu, eta, sample_n, device, percentage):
    logging.info('calculate_interval_metric ... (d={}, mu={}, eta={}, sample_n={})'.format(d, mu, eta, sample_n))
    for i in epochs:
        model_path = '{}{}'.format(path_prefix, i)
        if not os.path.exists(model_path):
            continue
        model.load(model_path)
        picp, pinaw, cwc = model.eval_cwc(test_data=data_set, d=d, mu=mu, eta=eta, sample_n=sample_n, device=device)
        picp, pinaw, cwc = picp.cpu(), pinaw.cpu(), cwc.cpu()
        logging.info('[epoch{}, predicted interval]: PICP={}, PINAW={}, CWC={}'.format(i, picp, pinaw, cwc))
        print('[epoch{}, predicted interval]: PICP={}, PINAW={}, CWC={}'.format(i, picp, pinaw, cwc))

# -------------------------------- analyse std -------------------------------------------


def corelation_std_data(data_name, path_prefix, model: UNCDM, exer_info_summary, epoch=None):
    '''
    calculate the correlation between std_data and relevant log counts
    :return:
    '''
    logging.info('corelation_std_data() ...')
    # mask for knowledge concept
    e_kn_cnt = exer_info_summary[:, :knowledge_n] + exer_info_summary[:, knowledge_n:]
    mask = e_kn_cnt > 0

    epoch_range = range(50) if epoch is None else [epoch]
    for epoch_i in epoch_range:
        model_path = path_prefix + str(epoch_i)
        if not os.path.exists(model_path):
            continue
        model.load(model_path)
        with torch.no_grad():
            print('lambda:', F.softplus(model.net.lambda_1_eta_stu.data), F.softplus(model.net.lambda_2_eta_stu.data),
                  F.softplus(model.net.lambda_1_eta_exer), F.softplus(model.net.lambda_2_eta_exer))
            _, stu_std_model, stu_std_data, _, k_diff_std_model, _, e_disc_std_model, e_std_data = model.get_pars()
            stu_std = combine_std_(stu_std_data, stu_std_model).numpy()
            sp_stu = pd.Series(stu_std.reshape(-1)).corr(pd.Series(model.net.stu_cnt.cpu().numpy().reshape(-1)), method='spearman')

            k_diff_std = combine_std_(e_std_data, k_diff_std_model)[mask].numpy().reshape(-1)
            e_disc_std = combine_std_(e_std_data, e_disc_std_model).numpy()
            exer_cnt = model.net.exer_cnt.cpu().numpy().reshape(-1)
            sp_diff = pd.Series(k_diff_std).corr(pd.Series(e_kn_cnt[mask].numpy()), method='spearman')
            sp_disc = pd.Series(e_disc_std.reshape(-1)).corr(pd.Series(exer_cnt), method='spearman')
            print(model_path, f'sp_stu={sp_stu}, sp_diff={sp_diff}, sp_disc={sp_disc}')
            logging.info(f'{model_path}, sp_stu={sp_stu}, sp_diff={sp_diff}, sp_disc={sp_disc}')


def analyse_std_model1(path_prefix, model: UNCDM, epoch=None, device='cpu'):
    '''
    :param path_prefix:
    :param model:
    :param epoch:
    :return:
    '''
    global train_set
    logging.info('analyse_std_model1() ...')

    epoch_range = range(50) if epoch is None else [epoch]
    for epoch_i in epoch_range:
        model_path = path_prefix + str(epoch_i)
        if not os.path.exists(model_path):
            continue
        model.load(model_path)
        with torch.no_grad():
            ret = model.analysis_std_model1(train_set, device)
            print(f'epoch {epoch_i}, corr_sum={ret[0]}')
            logging.info(f'epoch {epoch_i}, corr_sum={ret[0]}')


if __name__ == '__main__':
    model_name = 'UNeuralCDM'
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='FrcSub', help='data name')
    parser.add_argument('--sample-n', type=int, default=5, help='the number of sample trials')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--net-klw', type=float, default=0.01)
    parser.add_argument('--diag-klw', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--combine', type=str, default='x')  # x or +
    args = parser.parse_args()

    data_name = args.data
    combine_method = args.combine
    dst_folder = 'result/{}/{}'.format(data_name, model_name)
    if not os.path.isdir(dst_folder):
        os.makedirs(dst_folder)
    prefix = '{}/{}-s{}nkl{}dkl{}cb{}'.format(dst_folder, model_name, args.sample_n, args.net_klw, args.diag_klw, combine_method)
    logging.basicConfig(filename=prefix + '.log', level=logging.INFO, format='%(asctime)s %(message)s')

    stu_n, exer_n, knowledge_n, train_set, valid_set, test_set, stu_info_summary, exer_info_summary, _ = load_data_info(args.data)
    uncd = UNCDM(exer_n=exer_n, student_n=stu_n, knowledge_n=knowledge_n, stu_info_summary=stu_info_summary,
                exer_info_summary=exer_info_summary, kn_info_summary=None)
    model_prefix = prefix + '.snapshot'

    uncd.train(train_set, valid_set, test_set, lr=args.lr, device=args.device, epoch_n=args.epoch, sample_n=args.sample_n, net_klw=args.net_klw, diag_klw=args.diag_klw)
    calculate_interval_metric(test_set, uncd, model_prefix, range(100), d=1.96, mu=0.95, eta=10, sample_n=50, device=args.device, percentage=True)
    corelation_std_data(data_name, model_prefix, uncd, exer_info_summary, epoch=None)
    analyse_std_model1(model_prefix, uncd, None, device=args.device)

