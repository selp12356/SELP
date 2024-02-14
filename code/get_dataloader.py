import json
import os
import torch.nn as nn
import torch
import numpy as np
import torch.utils.data as data
from torch.utils.data import Dataset
import tqdm
import pandas as pd


def get_data(path):
    result = []
    with open(path, 'r', encoding = 'UTF-8') as src:
        for line in src:
            lines = json.loads(line)
            result.append(lines)

    return result

class myDataset(Dataset):
    def __init__(self,filepath,name):


        self.df = pd.DataFrame(self.index_subset(filepath,name))

        self.df = self.df.assign(id=self.df.index.values)

        self.unique_characters = sorted(self.df['class_name'].unique())
        self.num_classes = len(self.unique_characters)
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes)}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        self.datasetid_to_class_id = self.df.to_dict()['class_id']

    def __getitem__(self, item):

        label = self.datasetid_to_class_id[item]

        # print(label)
        # text = item['text']

        text = self.df['text'][item]

        return text, label

    def __len__(self):
        return len(self.df)

    def index_subset(self,path,name):
        texts=[]
        print(f'From {path} get data...')
        if name in ["huffpost","amazon","reuters","20news"]:
            datas = get_data(path)
            for line in datas:
                texts.append({
                    'text': line['text'],
                    'class_name': line['class_name']
                })
            return texts
        else:
            datas = get_data(path)
            for line in datas:
                texts.append({
                    'text': line['text'],
                    'class_name': line['label']
                })
            return texts

class CategoriesSampler():
    def __init__(self, data, n_batch, n_way , k_shot, q_shot):

        self.dataset = data
        self.n_batch = n_batch   #episode
        self.n_way = n_way
        self.k_shot = k_shot
        self.qshot = q_shot

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):

            support_set= []
            query_set=[]

            classes = np.random.choice(self.dataset.df['class_id'].unique(), size=self.n_way, replace=False)
            df = self.dataset.df[self.dataset.df['class_id'].isin(classes)]
            support_n = {n: None for n in classes}
            labels = []
            for n in classes:
                support  = df[df['class_id'] == n].sample(self.k_shot)
                support_n[n] = support

                for i, s in support.iterrows():
                    temp = {"text": s['text'], "label": s['class_name']}
                    support_set.append(temp)
                    labels.append(s['class_name'])

            for n in classes:
                query = df[(df['class_id'] == n) & (~df['id'].isin(support_n[n]['id']))].sample(self.qshot, replace=True)
                for i, q in query.iterrows():
                    temp = {"text": q['text'], "label": q['class_name']}
                    query_set.append(temp)

            yield np.stack(support_set), np.stack(query_set), labels



def get_loader(args):

    train_dataset = myDataset(args.train_path,args.dataset_name)

    train_sampler = CategoriesSampler(train_dataset,args.episodes,args.nway,args.kshot,args.qshot)
    val_data = myDataset(args.dev_path, args.dataset_name)
    val_sampler = CategoriesSampler(val_data,
                                    args.episodes,
                                    args.nway,
                                    args.kshot,
                                    args.qshot
                                    )
    test_data = myDataset(args.test_path, args.dataset_name)
    test_sampler = CategoriesSampler(test_data,
                                     args.episodes_q,
                                     args.nway,
                                     args.kshot,
                                     args.q_qshot
                                     )

    return train_sampler, val_sampler, test_sampler







