import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score,f1_score,recall_score, accuracy_score, roc_auc_score
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import math
import json


def euclidean_dist(x, y):
    '''
        Compute euclidean distance between two tensors
        '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def count_eu_dist(x,y):
    distance = torch.norm(x,y,dim=1,p=2)
    return distance

def center_loss(x,prototypes,q):
    """
    Args:
            x: feature matrix with shape (batch_size, feat_dim).
            prototypes: (nway, dim)
            q: query num
    """
    # 先把prototypes展开一下
    repeated_tensor = prototypes.repeat(1, q)
    repeated_tensor = repeated_tensor.view(5*q,-1)
    batch_size = repeated_tensor.size()[0]

    dist = torch.pow(x-repeated_tensor, 2).sum()
    dist = math.sqrt(dist)
    dist = dist/2.0/batch_size

    return dist


def constrastive_loss(features, labels, temperature, mask=None):
    features = F.normalize(features, p=2, dim=1)

    batch_size = features.shape[0]
    labels = labels.contiguous().view(-1, 1)


    mask = torch.eq(labels, labels.T).float().cuda()

    # compute logits, anchor_dot_contrast: (bsz, bsz), x_i_j: (z_i*z_j)/t
    # (bsz, bsz), x_i_j: (z_i * z_j) / t
    anchor_dot_contrast = torch.div(
        torch.matmul(features, features.T),
        temperature)


    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()


    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).cuda(),
        0
    )

    mask = mask * logits_mask
    # _op = labels.argmax(dim=1)
    _mask = torch.ones_like(mask).float()
    for i,value in enumerate(labels):
        if value==5:
            _mask[i,:]=0.0
            _mask[:,i]=0.0
    _mask.cuda()
    mask = mask * _mask

    # compute log_prob

    exp_logits = torch.exp(logits) * logits_mask

    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    if torch.any(torch.isnan(log_prob)):
        raise ValueError("Log_prob has nan!")

    # compute mean of log-likelihood over positive

    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1)

    if torch.any(torch.isnan(mean_log_prob_pos)):
        raise ValueError("mean_log_prob_pos has nan!")

    # loss
    loss = - mean_log_prob_pos
    # print("SCL Loss:")
    # print(loss.shape)
    if torch.any(torch.isnan(loss)):
        raise ValueError("loss has nan!")
    loss = loss.mean()

    return loss


class Loss_Fn(torch.nn.Module):
    def __init__(self, args):
        super(Loss_Fn, self).__init__()
        self.args = args
        self.loss_ce = CrossEntropyLoss()
        self.hdim = 768
        self.loss_weight = nn.Parameter(torch.ones(3))
        self.w = nn.Parameter(torch.ones(2))

    def cul_proto(self,support_emb_cls, prompts):

        prototypes = support_emb_cls.view(10, -1, support_emb_cls.shape[1])  # N X K X dim
        prototypes = torch.mean(prototypes, dim=1)  # 在每个类上求均值[n, dim]

        # w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        # w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        w_proto = self.args.alpha * prototypes + (1 - self.args.alpha) * prompts

        return w_proto



    def forward(self, support_emb_cls,query_emb_cls,prompts, query_lcm,loss_simcse,labels_ids, flag, modee="test"):
        if modee=="test":
            query_size = self.args.nway * self.args.q_qshot
        else:
            query_size = self.args.nway * self.args.qshot
        support_size = self.args.nway * self.args.kshot


        prototypes = support_emb_cls.view(self.args.nway, -1, support_emb_cls.shape[1])  # N X K X dim
        prototypes = torch.mean(prototypes, dim=1)  # 在每个类上求均值[n, dim]



        # w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        # w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        # w_proto = w1 * prototypes + w2 * prompts
        w_proto = self.args.alpha * prototypes + (1 - self.args.alpha) * prompts

        #center loss
        if modee=="test":
            qshot = self.args.q_qshot
        else:
            qshot = self.args.qshot
        center_loss_num = center_loss(query_emb_cls, w_proto, qshot)


        # calculate ptototypical loss
        dists = euclidean_dist(query_emb_cls, w_proto)


        log_p_y = F.log_softmax(-dists, dim=1).cuda()  # num_query x num_class

        query_labels = labels_ids[support_size:]
        query_labels = torch.tensor(query_labels, dtype=float).cuda()
        # dists = torch.tensor(dists, dtype=float).cuda()
        #normal loss
        loss = - query_labels * log_p_y
        loss = loss.mean()

        #KL LOSS

        kl_loss = F.kl_div(log_p_y.float(), query_lcm.float(), reduction='batchmean').float()
        # totalloss = loss + self.args.gama * loss_simcse

        # totalloss = self.args.gama * loss_simcse + self.args.gama1 * kl_loss + self.args.gama2 * center_loss_num

        loss_list = [kl_loss, center_loss_num]
        final_loss = []
        for i in range(len(loss_list)):
            final_loss.append(loss_list[i] / (2 * self.loss_weight[i].pow(2)) + torch.log(self.loss_weight[i]))
        all_loss = torch.sum(torch.stack(final_loss))



        x, _ = torch.max(log_p_y, dim=1, keepdim=True)
        one = torch.ones_like(log_p_y)
        zero = torch.zeros_like(log_p_y)
        y_pred = torch.where(log_p_y >= x, one, log_p_y)
        y_pred = torch.where(log_p_y < x, zero, y_pred)   #这个地方就处理成了0 1向量

        target_mode = 'macro'

        # query_labels = query_labels.cpu()
        # y_pred = y_pred.cpu()

        # if self.args.losstype == 1:
        #     loss1 = self.loss_ce(-dists, query_labels)
        # else:
        #     loss1 = total_loss

        sq = query_labels.cpu().detach()
        yp = y_pred.cpu().detach()

        # print("----------------------------------------------------------------------------------")
        # print(sq)
        # print(yp)

        p = precision_score(sq, yp, average=target_mode)
        r = recall_score(sq, yp, average=target_mode)
        f = f1_score(sq, yp, average=target_mode)
        acc = accuracy_score(sq, yp)

        y_score = log_p_y
        y_score = y_score.cpu().detach()
        auc = roc_auc_score(sq, y_score)

        return all_loss, p,r,f,acc,auc

