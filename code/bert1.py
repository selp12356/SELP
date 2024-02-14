import numpy
import torch
import torch.nn as nn
import random
from transformers import BertTokenizer, BertConfig, BertModel,AutoTokenizer,AutoModel
from transformers import BertLayer
import numpy as np
from get_loss2 import Loss_Fn
import torch.nn.functional as F
import copy
import json


class Bert_pure(nn.Module):

    def __init__(self, args):
        super(Bert_pure, self).__init__()
        self.args = args

        # self.tokenizer = BertTokenizer.from_pretrained(self.args.filevocab, do_lower_case = False)
        # config = BertConfig.from_json_file(self.args.fileModelConfig)  #配置文件
        # self.bert = BertModel.from_pretrained(self.args.fileModel, config = config)

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.filevocab)
        self.bert = AutoModel.from_pretrained(self.args.fileModel)

        if self.args.numFreeze > 0:
            self.freeze_layers(self.args.numFreeze)
        #




    def freeze_layers(self, numFreeze):
        unfreeze_layers = []
        for i in range(numFreeze, 12):
            unfreeze_layers.append("layer."+str(i))

        for name, param in self.bert.named_parameters():   #取出某一层的特征
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break

    def forward(self, text, modd):


        input_ids=[]
        max_len=0
        if modd == "text":
            max_len = self.args.text_max_len
        else:
            max_len = self.args.label_max_len

        text_ids = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_len
        )
        text_ids = text_ids["input_ids"]

        bs = len(text)
        text_ids = torch.tensor(text_ids)

        text_len = text_ids.size(1)

        atten_mask_text = torch.ones(bs, text_len)
        atten_mask_text[text_ids == 0] = 0


        text_ids = text_ids.cuda()
        atten_mask_text = atten_mask_text.cuda()


        output_text = self.bert(
            text_ids,
            attention_mask = atten_mask_text,
            output_hidden_states=True
        )

        every_layer = output_text[2]

        layer_output = every_layer[11]

        return layer_output


    def get_last_layer(self):

        return self.bert.encoder.layer[11]




class MultiPromptLearner(nn.Module):
    def __init__(self, args):
        super(MultiPromptLearner, self).__init__()
        self.args = args
        # prompt pool size -> (pool_len,prompt_len,hidden_size)
        # key size -> (pool_len,key_hidden_size)
        # prompt length
        self.prompt_len = self.args.prompt_len
        self.hidden_size = self.args.hidden_size
        self.key_hidden_size = self.args.key_hidden_size  #768
        self.pool_len = self.args.pool_len
        #key的初始化方式
        prompt_key_init = self.args.key_init_method

        # initializing prompt key
        # prompt_key_init = 'uniform'
        # key size -> (pool_len,key_hidden_size)
        key_shape = (self.pool_len, self.key_hidden_size)
        if prompt_key_init == 'zero':
            self.prompt_key = nn.Parameter(torch.zeros(key_shape))
        elif prompt_key_init == 'uniform':
            self.prompt_key = nn.Parameter(torch.randn(key_shape))
            nn.init.uniform_(self.prompt_key, -1, 1)
        elif prompt_key_init == 'normal':
            self.prompt_key = nn.Parameter(torch.randn(key_shape))
            nn.init.normal_(self.prompt_key, std=0.02)

        # prompt pool size -> (pool_len,prompt_len,hidden_size)
        print("Initializing a generic context")
        ctx_vectors = torch.empty(self.pool_len, self.prompt_len, self.hidden_size, dtype=torch.float)
        nn.init.normal_(ctx_vectors, std=0.02)

        self.prompt = nn.Parameter(ctx_vectors)  # to be optimized

    def forward(self, name_embs):
        # name_embs_cls size (n_cls,768)
        name_embs_cls = name_embs[:, :1, :].squeeze()
        prefix = name_embs[:, :1, :]
        suffix = name_embs[:, 1:, :]
        prompt_key = self.prompt_key
        name_embs_cls = name_embs_cls / name_embs_cls.norm(dim=-1, keepdim=True)
        prompt_key = prompt_key / prompt_key.norm(dim=-1, keepdim=True)

        # calculate weights
        weights = name_embs_cls @ prompt_key.T  # n_cls, pool_len
        weights = F.softmax(weights, dim=1)  # n_cls, pool_len
        prompt = self.prompt.permute(2, 0, 1)  # dim, pool_len, prompt_len
        weights = weights.unsqueeze(dim=0)  # 1, n_cls, pool_len
        weighted_prompts = torch.matmul(weights, prompt)  # dim, n_cls, prompt_len
        weighted_prompts = weighted_prompts.permute(1, 2, 0)  # n_cls, prompt_len, dim

        # print("weight_prompts_size----------")
        # print(weighted_prompts.size())
        # print("name_emb_size------------------------")
        # print(name_embs_cls.size())


        prompts = torch.cat(
                [
                    prefix,
                    weighted_prompts,  # (n_cls, prompt_len, dim)
                    suffix  # (n_cls, name_len, dim)
                ],
                dim=1,
        )


        return prompts


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class myModel(nn.Module):
    def __init__(self, args):
        super(myModel, self).__init__()
        self.args = args
        self.bert = Bert_pure(self.args)
        self.last_layer = self.bert.get_last_layer()
        self.meta_prompt = MultiPromptLearner(self.args)
        self.loss_fn = Loss_Fn(args)
        self.eps = 1e-6
        self.inff = 1e12
        self.temprature = 0.05

        self.linear = nn.Linear(self.args.nway, self.args.nway)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.1)  # follow the default in bert model

        # self.W_q = nn.Parameter(torch.randn(768, 768))
        # self.W_k = nn.Parameter(torch.randn(768, 768))

        self.BN = nn.BatchNorm1d(num_features=768)
        self.sim = Similarity(self.args.temprature)

        # self.lgm = L_GM_loss.LGMLoss(5, 768)

    def loss_ce(self, logits, Y):
        loss = nn.CrossEntropyLoss()
        output = loss(logits, Y)
        return output

    def calculateDropoutCLLoss(self, embedding):
        batch_size = embedding.shape[0]
        embedding = embedding.unsqueeze(1).repeat(1, 2, 1).view(batch_size * 2, -1)

        CLSEmbedding = self.BN(embedding)
        batchEmbedding = CLSEmbedding.view((batch_size, 2, 768))  # (bs, num_sent, hidden)

        # Separate representation
        z1, z2 = batchEmbedding[:, 0], batchEmbedding[:, 1]

        z2 = self.dropout(z2)

        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        logits = cos_sim

        labels = torch.arange(logits.size(0)).long().cuda()
        lossVal = self.loss_ce(logits, labels)

        return lossVal


    def forward(self, text, labels, labels_ids, flag, unique_list, modee):
        support_size = self.args.nway * self.args.kshot

        text_emb = self.bert(text,"text")   #[b,seq_len, 768]
        label_emb = self.bert(unique_list,"label")
        # label_emb = self.bert()  # [5,label_len,768]
        # text_emb2 = self.bert(text, "text")  #[b,seq_len, 768]


        prompts = self.meta_prompt(label_emb)

        text_emb_new = self.last_layer(text_emb)
        label_emb_new = self.last_layer(prompts)

        text_emb_new_cls = text_emb_new[0][:, :1, :].squeeze()  # [bs,768]
        label_emb_new_cls = label_emb_new[0][:, :1, :].squeeze()  # [5,768]



        support_emb_cls = text_emb_new_cls[:support_size]
        query_emb_cls = text_emb_new_cls[support_size:]



        #LCM DOT
        dot_sim = torch.matmul(query_emb_cls, label_emb_new_cls.T)  # [query_size, nway]
        dot_sim_mlp = self.linear(dot_sim)
        dot_sim_mlp = self.softmax(dot_sim_mlp)


        query_onehot = labels_ids[support_size:]
        query_onehot_tensor = torch.tensor(query_onehot,dtype=float).cuda() #[query_size, nway]
        query_onehot_ = self.args.beta * query_onehot_tensor + dot_sim_mlp
        query_onehot_ = self.softmax(query_onehot_)

        # loss_simcse = 0
        # query_onehot_ = 0
        loss_simcse =0


        loss,p,r,f,acc,auc= self.loss_fn(support_emb_cls,query_emb_cls,label_emb_new_cls, query_onehot_,loss_simcse,labels_ids,flag,modee)

        return loss,p,r,f,acc,auc







