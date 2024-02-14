import argparse
from argparse import ArgumentParser
import os
import json
from get_args import Args
import torch
import numpy as np
import random
import utils
# linux
from bert1 import myModel
# from Bert_mid import myModel
from transformers import AdamW, get_linear_schedule_with_warmup
import get_dataloader
import time
import tqdm
import warnings
warnings.filterwarnings("ignore")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def write_config(cf_path, save_path):
    with open(cf_path, 'r') as f:
        data = json.load(f)

    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)

def  init_optim(args, model):

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [ p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    return optimizer

def init_lr_scheduler(args, optim):

    t_total = args.epochs * args.episodes
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    return scheduler

def deal_data(support_set, query_set, episode_labels):

    text, labels, flag = [], [], []

    new_list = []

    for i in episode_labels:
        if i in new_list:
            pass
        else:
            new_list.append(i)


    for id, i in enumerate(new_list):
        for x in support_set:
            if (x["label"] == i):
                text.append(x["text"])
                labels.append(x["label"])
                flag.append(id)
    for id, i in enumerate(new_list):
        for x in query_set:
            if (x["label"] == i):
                text.append(x["text"])
                labels.append(x["label"])
                flag.append(id)

    label_ids = []
    for label in labels:
        tmp = []
        for l in new_list:
            if l == label:
                tmp.append(1)
            else:
                tmp.append(0)
        label_ids.append(tmp)

    return text, labels, label_ids, flag, new_list

def test(args, test_dataloader, model, modee, config_name):
    val_p = []
    val_r = []
    val_loss = []
    val_f1 = []
    val_acc = []
    val_auc = []
    logger = args.logger
    with torch.no_grad():
        model.eval()

        # for batch in test_dataloader:
        for i, batch in enumerate(test_dataloader):

            if (i % 100 == 0):
                logger.info(f'---now test {i} step-------')
                print(f'---now test {i} step-------')
            support_set, query_set, episode_labels = batch

            text, labels, labels_ids, flag, unique_list  = deal_data(support_set, query_set, episode_labels)

            loss, p, r, f, acc, auc = model(text, labels, labels_ids, flag,unique_list, modee="test")
            val_loss.append(loss.item())
            val_acc.append(acc)
            val_p.append(p)
            val_r.append(r)
            val_f1.append(f)
            val_auc.append(auc)


        avg_loss = np.mean(val_loss)
        avg_acc = np.mean(val_acc)
        avg_p = np.mean(val_p)
        avg_r = np.mean(val_r)
        avg_f1 = np.mean(val_f1)
        avg_auc = np.mean(val_auc)


        print('Test p: {}'.format(avg_p))
        print('Test r: {}'.format(avg_r))
        print('Test f1: {}'.format(avg_f1))
        print('Test acc: {}'.format(avg_acc))
        print('Test auc: {}'.format(avg_auc))
        print('Test Loss: {}'.format(avg_loss))


        logger.info('Test p: {}'.format(avg_p))
        logger.info('Test r: {}'.format(avg_r))
        logger.info('Test f1: {}'.format(avg_f1))
        logger.info('Test acc: {}'.format(avg_acc))
        logger.info('Test auc: {}'.format(avg_auc))
        logger.info('Test Loss: {}'.format(avg_loss))


        path = args.save_path

        if not os.path.exists(path):
            os.makedirs(path)

        path = path + config_name
        with open(path, "a+") as fout:
            tmp = {"modee":modee,"p": avg_p, "r": avg_r, "f1": avg_f1, "acc": avg_acc, "auc": avg_auc, "Loss": avg_loss}
            fout.write("%s\n" % json.dumps(tmp, ensure_ascii=False))



def train(args, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):

    if val_dataloader is None:
        acc_best_state = None
        f1_best_state = None
    train_loss, epoch_train_loss = [], []

    train_acc, epoch_train_acc = [], []
    train_p, epoch_train_p = [], []
    train_r, epoch_train_r = [], []
    train_f1, epoch_train_f1 = [], []
    train_auc, epoch_train_auc = [], []

    val_loss, epoch_val_loss = [], []
    val_acc, epoch_val_acc = [], []
    val_p, epoch_val_p = [], []
    val_r, epoch_val_r = [], []
    val_f1, epoch_val_f1 = [], []
    val_auc, epoch_val_auc = [], []
    best_p = 0
    best_r = 0
    best_f1 = 0
    best_acc = 0
    best_auc = 0
    best_p_s = 0
    best_r_s = 0
    best_f1_s = 0
    best_acc_s = 0
    best_auc_s = 0
    logger = args.logger

    p_best_model_path = args.save_path + '/' + args.dataset + '_p_best_model.pth'
    r_best_model_path = args.save_path + '/' + args.dataset + '_r_best_model.pth'
    f1_best_model_path = args.save_path + '/' + args.dataset + '_f1_best_model.pth'
    acc_best_model_path = args.save_path + '/' + args.dataset + '_acc_best_model.pth'
    auc_best_model_path = args.save_path + '/' + args.dataset + '_auc_best_model.pth'

    for epoch in range(args.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        logger.info('=== Epoch: {} ==='.format(epoch))
        print('=== Here: {} ==='.format(args.checkpoint_path))
        model.train()

        for i, batch in enumerate(tr_dataloader):
            optim.zero_grad()

            support_set, query_set, episode_labels = batch

            text,labels, labels_ids, flag, unique_list = deal_data(support_set, query_set, episode_labels)

            loss, p,r,f,acc,auc = model(text, labels,labels_ids, flag,unique_list, modee="train")  #support_embeddings, query_embeddings


            loss.backward()
            optim.step()
            lr_scheduler.step()
            if(i%100==0):
                print(f'---now {i} step-------')
                logger.info(f'---now {i} step-------')
            train_loss.append(loss.item())
            train_p.append(p)
            train_r.append(r)
            train_f1.append(f)
            train_acc.append(acc)
            train_auc.append(auc)

        avg_loss = np.mean(train_loss[-args.episodes:])

        avg_acc = np.mean(train_acc[-args.episodes:])
        avg_p = np.mean(train_p[-args.episodes:])
        avg_r = np.mean(train_r[-args.episodes:])
        avg_f1 = np.mean(train_f1[-args.episodes:])
        avg_auc = np.mean(train_auc[-args.episodes:])

        print('Avg Train Loss: {}, Avg Train p: {}, Avg Train r: {}, Avg Train f1: {}, Avg Train acc: {}, Avg Train auc: {}'.format(avg_loss, avg_p, avg_r, avg_f1, avg_acc, avg_auc))
        logger.info('Avg Train Loss: {}, Avg Train p: {}, Avg Train r: {}, Avg Train f1: {}, Avg Train acc: {}, Avg Train auc: {}'.format(avg_loss, avg_p, avg_r, avg_f1, avg_acc, avg_auc))

        epoch_train_loss.append(avg_loss)
        epoch_train_acc.append(avg_acc)
        epoch_train_p.append(avg_p)
        epoch_train_r.append(avg_r)
        epoch_train_f1.append(avg_f1)
        epoch_train_auc.append(avg_auc)

        if val_dataloader is None:
            continue
        with torch.no_grad():
            model.eval()

            # for batch in val_dataloader:
            for i, batch in enumerate(val_dataloader):

                if(i%100==0):
                    print(f'---now val {i} step-------')
                    logger.info(f'---now val {i} step-------')


                support_set, query_set, episode_labels = batch
                text, labels, labels_ids, flag , unique_list= deal_data(support_set, query_set, episode_labels)
                loss, p, r, f, acc, auc= model(text, labels,labels_ids, flag,unique_list, modee="valid")

                val_loss.append(loss.item())

                val_acc.append(acc)
                val_p.append(p)
                val_r.append(r)
                val_f1.append(f)
                val_auc.append(auc)
                # print(f"Episode: {episode}: Loss: {loss.item()}, Acc: {acc.item()}, F1: {f1}")

            avg_loss = np.mean(val_loss[-args.episodes:])
            avg_acc = np.mean(val_acc[-args.episodes:])
            avg_p = np.mean(val_p[-args.episodes:])
            avg_r = np.mean(val_r[-args.episodes:])
            avg_f1 = np.mean(val_f1[-args.episodes:])
            avg_auc = np.mean(val_auc[-args.episodes:])

            # print('Avg Train Loss: {}, Avg Train Precision: {}, Avg Train Recall: {},Avg Train F1: {}'.format(avg_loss, avg_p, avg_r, avg_f1))
            epoch_val_loss.append(avg_loss)
            epoch_val_acc.append(avg_acc)
            epoch_val_p.append(avg_p)
            epoch_val_r.append(avg_r)
            epoch_val_f1.append(avg_f1)
            epoch_val_auc.append(avg_auc)


        p_prefix = ' (Best)' if avg_p >= best_p else ' (Best:{})'.format(best_p)
        r_prefix = ' (Best)' if avg_r >= best_r else ' (Best: {})'.format(best_r)
        f1_prefix = ' (Best)' if avg_f1 >= best_f1 else ' (Best: {})'.format(best_f1)
        acc_prefix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(best_acc)
        auc_prefix = ' (Best)' if avg_auc >= best_auc else ' (Best: {})'.format(best_auc)

        print(
            'Avg Val Loss: {}, Avg Val p: {}{}, Avg Val r: {}{}, Avg Val f1: {}{}, Avg Val acc: {}{}, Avg Val auc: {}{}'.format(
                avg_loss, avg_p, p_prefix, avg_r, r_prefix, avg_f1, f1_prefix, avg_acc, acc_prefix, avg_auc, auc_prefix))

        logger.info('Avg Val Loss: {}, Avg Val p: {}{}, Avg Val r: {}{}, Avg Val f1: {}{}, Avg Val acc: {}{}, Avg Val auc: {}{}'.format(
                avg_loss, avg_p, p_prefix, avg_r, r_prefix, avg_f1, f1_prefix, avg_acc, acc_prefix, avg_auc, auc_prefix))

        if avg_p >= best_p:
            torch.save(model.state_dict(), p_best_model_path)
            best_p = avg_p
            best_p_s = epoch
            p_best_state = model.state_dict()

        # if avg_r >= best_r:
        #     torch.save(model.state_dict(), r_best_model_path)
        #     best_r = avg_r
        #     best_r_s = epoch
        #     r_best_state = model.state_dict()
        #
        # if avg_f1 >= best_f1:
        #     torch.save(model.state_dict(), f1_best_model_path)
        #     best_f1 = avg_f1
        #     best_f1_s = epoch
        #     f1_best_state = model.state_dict()

        if avg_acc >= best_acc:
            torch.save(model.state_dict(), acc_best_model_path)
            best_acc = avg_acc
            best_acc_s = epoch
            acc_best_state = model.state_dict()

        # if avg_auc >= best_auc:
        #     torch.save(model.state_dict(), auc_best_model_path)
        #     best_auc = avg_auc
        #     best_auc_s = epoch
        #     auc_best_state = model.state_dict()

        last_p = epoch - best_p_s
        last_r = epoch - best_r_s
        last_f1 = epoch - best_f1_s
        last_acc = epoch - best_acc_s
        last_auc = epoch - best_auc_s
        s_list = [last_p, last_r, last_f1, last_acc, last_auc]
        last_change = max(s_list)
        if last_change >= 50:
            break

    for name in ['epoch_train_loss', 'epoch_train_p', 'epoch_train_r', 'epoch_train_f1', 'epoch_train_acc',
                 'epoch_train_auc', 'epoch_val_loss', 'epoch_val_p', 'epoch_val_r', 'epoch_val_f1', 'epoch_val_acc',
                 'epoch_val_auc']:
        utils.save_list_to_file(os.path.join(args.save_path,
                                       args.dataset + name + '.txt'), locals()[name])

    return p_best_state
    # return epoch_train_loss
if __name__ == '__main__':
    parser = ArgumentParser()
    #linux
    parser.add_argument('--config', default="../sh/huffpost.json", type=str)
    # parser.add_argument('--config', default="./config/huffpost_linux_3.json", type=str)
    # parser.add_argument('--config', default="./config/banking77_linux.json", type=str)
    # parser.add_argument('--config', default="./config/banking77.json", type=str)
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--kshot', default=-1, type=int)
    parser.add_argument('--beta', default=-1.0, type=float)
    parser.add_argument('--temprature', default=-1.0, type=float)
    parser.add_argument('--tempra1', default=-1.0, type=float)
    parser.add_argument('--dataset_num', default="09", type=str)
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--qshot', default=-1, type=int)
    parser.add_argument('--alpha', default=-1.0, type=float)
    parser.add_argument('--gama', default=-1.0, type=float)
    parser.add_argument('--output', default="no", type=str)
    parser.add_argument('--se_layer', default=-1, type=int)
    parser.add_argument('--weight_decay', default=0.00, type=float)
    parser.add_argument('--margin', default=-1.0, type=float)
    parser.add_argument('--prompt_len', default=-1, type=int)
    parser.add_argument('--numFreeze', default=-1, type=int)
    parser.add_argument('--pool_len', default=-1, type=int)
    parser.add_argument('--epochs', default=-1, type=int)
    parser.add_argument('--warmup_steps', default=-1, type=int)
    parser.add_argument('--step', default=-1, type=int)
    parser.add_argument('--learning_rate', default=-1.0, type=float)
    parser.add_argument('--dropout', default=-1.0, type=float)

    parser.add_argument('--gama1', default=-1.0, type=float)
    parser.add_argument('--gama2', default=-1.0, type=float)

    parser.add_argument('--dataset_name', default="no", type=str)
    parser.add_argument('--task', default="no", type=str)
    parser.add_argument('--text_len', default=-1, type=int)
    parser.add_argument('--label_len', default=-1, type=int)

    parser.add_argument('--optionn', default="bert_layer", type=str,help="mlp,nothing,bert_layer,mean")
    # parser.add_argument('--output', default="no", type=str)
    # parser.add_argument('--seed', default=-1, type=int)

    # parser.add_argument('--gpus', type=int, default=3)
    args_ = parser.parse_args()
    args = Args(args_.config)
    if args_.gpu != -1:
        args.gpu = args_.gpu
    if args_.tempra1 != -1.0:
        args.temprature1 = args_.tempra1
    if args_.kshot != -1:
        args.kshot = args_.kshot
    if args_.beta != -1.0:
        args.beta = args_.beta
    if args_.temprature != -1.0:
        args.temprature = args_.temprature
    if args_.dataset_num != "09":
        args.dataset = args_.dataset_num
    if args_.seed != -1:
        args.seed = args_.seed
    if args_.qshot != -1:
        args.qshot = args_.qshot
    if args_.alpha != -1.0:
        args.alpha = args_.alpha
    if args_.gama != -1.0:
        args.gama = args_.gama
    if args_.se_layer != -1:
        args.se_layer = args_.se_layer
    if args_.weight_decay != 0.00:
        args.weight_decay = args_.weight_decay
    if args_.margin != -1.0:
        args.margin = args_.margin
    if args_.prompt_len != -1:
        args.prompt_len = args_.prompt_len
    if args_.numFreeze != -1:
        args.numFreeze = args_.numFreeze
    if args_.pool_len != -1:
        args.pool_len = args_.pool_len
    if args_.epochs != -1:
        args.epochs = args_.epochs
    if args_.warmup_steps != -1:
        args.warmup_steps = args_.warmup_steps
    if args_.step != -1:
        args.step = args_.step
    if args_.learning_rate != -1.0:
        args.learning_rate = args_.learning_rate
    if args_.dropout != -1.0:
        args.dropout = args_.dropout
    if args_.text_len != -1:
        args.text_max_len = args_.text_len
    if args_.label_len != -1:
        args.label_max_len = args_.label_len
    if args_.gama1 != -1.0:
        args.gama1 = args_.gama1
    if args_.gama2 != -1.0:
        args.gama2 = args_.gama2
    args.optionn = args_.optionn
    args.dataset_name = args_.dataset_name

    args.train_path = args.train_path.format(args.dataset)
    args.dev_path = args.dev_path.format(args.dataset)
    args.test_path = args.test_path.format(args.dataset)
    # print(args.train_path)



    pathname = "../output_lcm_simcse/{}/test_{}_{}_{}".format(args_.dataset_name,args_.task, args_.output, time.strftime("%m-%d_%H-%M-%S"))
    if os.path.exists(pathname):
        pass
    else:
        os.makedirs(pathname)
    args.save_path = pathname

    temp_iden = "here_{}_{}_{}_gpu_{}".format(args_.dataset_name, args_.task, args_.output, args.gpu)
    args.checkpoint_path = temp_iden

    # print("----i am here 1---------")

    set_seed(args.seed)

    # logger = utils.get_logger(args.dataset,pathname)
    # logger.info(args)
    # args.logger = logger
    config_name = '/' + args_.dataset_num + '_config.json'
    save_config_path = args.save_path + config_name
    args.write_self(save_config_path)
    # write_config(args_.config,  save_config_path)
    # print("----i am here 2---------")
    args.show_self()

    # print("----i am here 3---------")

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    # print("----i am here 4---------")

    mymodel = myModel(args)
    mymodel.cuda()

    # print("----i am here 5---------")

    optim = init_optim(args, mymodel)
    lr_scheduler = init_lr_scheduler(args, optim)

    train_dataloader,dev_dataloader,test_dataloader = get_dataloader.get_loader(args)

    # print("----i am here 6---------")


    results = train(args=args,
                    tr_dataloader= train_dataloader,
                    model = mymodel,
                    optim= optim,
                    lr_scheduler = lr_scheduler,
                    val_dataloader=dev_dataloader
                    )

    p_best_model_path = args.save_path + '/' + args.dataset + '_p_best_model.pth'
    r_best_model_path = args.save_path + '/' + args.dataset + '_r_best_model.pth'
    f1_best_model_path = args.save_path + '/' + args.dataset + '_f1_best_model.pth'
    acc_best_model_path = args.save_path + '/' + args.dataset + '_acc_best_model.pth'
    auc_best_model_path = args.save_path + '/' + args.dataset + '_auc_best_model.pth'

    mymodel.load_state_dict(torch.load(p_best_model_path))
    print('Testing with p best model..')
    test(args=args,
         test_dataloader=test_dataloader,
         model=mymodel,
         modee='best_p',
         config_name=config_name)

    # mymodel.load_state_dict(torch.load(r_best_model_path))
    # print('Testing with r best model..')
    # test(args=args,
    #      test_dataloader=test_dataloader,
    #      model=mymodel,
    #      modee='best_r',
    #      config_name=config_name)
    #
    # mymodel.load_state_dict(torch.load(f1_best_model_path))
    # print('Testing with f1 best model..')
    # test(args=args,
    #      test_dataloader=test_dataloader,
    #      model=mymodel,
    #      modee='best_f1',
    #      config_name=config_name)

    # mymodel.load_state_dict(torch.load(acc_best_model_path))
    # print('Testing with acc best model..')
    # test(args=args,
    #      test_dataloader=test_dataloader,
    #      model=mymodel,
    #      modee='best_acc',
    #      config_name=config_name)
    #
    # mymodel.load_state_dict(torch.load(auc_best_model_path))
    # print('Testing with auc best model..')
    # test(args=args,
    #      test_dataloader=test_dataloader,
    #      model=mymodel,
    #      modee='best_auc',
    #      config_name=config_name)

    os.remove(p_best_model_path)
    # os.remove(r_best_model_path)
    # os.remove(f1_best_model_path)
    os.remove(acc_best_model_path)
    # os.remove(auc_best_model_path)



