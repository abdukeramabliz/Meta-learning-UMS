import sys

import torch
import numpy as np
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from load_data import *
from transformer import Encoder, WeiYunet
import scipy.stats
from torch.utils.data import DataLoader
import argparse

import torch.utils.data as Data

from meta import Meta


def mean_confidence_interval(accs, confidence=0.95):  # 置信区间，置信度为0.95
    n = accs.shape[0]  # n是accs数组中元素的个数
    m, se = np.mean(accs), scipy.stats.sem(accs)
    # mean（）求取均值
    # scipy.stats.sem(arr，axis = 0，ddof = 0)函数用于计算输入数据平均值的标准误差
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():
    # 设置种子的用意是一旦固定种子，后面依次生成的随机数其实都是固定的
    torch.manual_seed(222)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed_all(222)  ##为当前GPU设置随机种子
    np.random.seed(222)  # 设置随机种子

    print(args)  # 输出参数

    config = [

        ('flatten', []),
        ('linear', [128, 512]),
        ('leakyrelu', [0.2, True]),
        ('bn', [128]),
        ('linear', [32, 128]),
        ('leakyrelu', [0.2, True]),
        # ('bn', [32]),
        ('linear', [5, 32]),
        ('leakyrelu', [0.2, True])
    ]

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())  # 从参数tensor集合-- parameters中找到可变的tensor形成一个新的集合。
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)
    encoder = Encoder().to(device)
    global accs_train, tes_mask


    weiyu = WeiYunet('meta_train_support', 'meta_train_query', mode='train', k_shot=args.k_spt,
                     batchsz=200, n_way=args.n_way)
    weiyu_test = WeiYunet('meta_test_support', 'meta_test_query', mode='test', k_shot=args.k_spt,
                          batchsz=15, n_way=args.n_way)


    for epoch in range(args.epoch):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(weiyu, args.task_num, False, drop_last=True)

        for x_spts, y_spts, x_qrys, y_qry in db:
            x_spt_encs,x_qry_encs=[],[]
            x_spts,x_qrys = x_spts.to(device),x_qrys.to(device)
            for x_spt in x_spts:
                x_spt_enc, x_spt_self_attns = encoder(x_spt)
                x_spt_encs.append(x_spt_enc)
            x_spt_enc = torch.stack(x_spt_encs, 0)

            for x_qry in x_qrys:
                x_qry_enc, x_qry_self_attns = encoder(x_qry)
                x_qry_encs.append(x_qry_enc)
            x_qry_enc = torch.stack(x_qry_encs, 0)
            x_spt_enc, y_spts, x_qry_enc, y_qry = x_spt_enc.to(device), y_spts.to(device), x_qry_enc.to(device), y_qry.to(device)

            # print(y_spt.type(), x_spt_enc.size())
            #x_spt_enc.size[2, 5, 35, 512] y_spt [2,5,35] x_qry_enc[2, 1, 35, 512], y_qry[2, 1, 35],y_qry_mask[2, 1, 35]

            for n in range(args.k_spt):
                x_spt =x_spt_enc[:,n]
                y_spt =y_spts[:,n]
                x_qry =x_qry_enc.squeeze(1)
                y_qry =y_qry.squeeze(1)
                y_qry_mask = torch.tensor(y_qry, dtype=bool).to(device)
                accs = maml(x_spt, y_spt, x_qry, y_qry,y_qry_mask)

            # if step % 4 == 0:
        print('epoch:', epoch ,'\ttraining acc:', accs)
        with open('log1.txt', 'a', encoding='utf-8') as f:
            f.write('train' + ' ' + str(epoch) + '\t' + str(accs) + '\n')

        if epoch % 3 == 0:  # evaluation

            db_test = DataLoader(weiyu_test, args.n_way, False, drop_last=True)

            accs_all_test, recall_all, f1_all, pres_all = [], [], [], []

            for te_x_spt, te_y_spts, te_x_qry, te_y_qry in db_test:

                te_x_spt, te_y_spts, te_x_qry, te_y_qry = te_x_spt.to(device), te_y_spts.to(device), \
                                                          te_x_qry.squeeze(1).to(device), te_y_qry.squeeze(1).to(device)
                # print(te_x_qry)
                te_x_spt_encs =  []
                for x_spt in te_x_spt:
                    te_x_spt_enc, te_x_spt_self_attns = encoder(x_spt)
                    te_x_spt_encs.append(te_x_spt_enc)
                te_x_spt_enc = torch.stack(te_x_spt_encs, 0)
                te_x_qry_enc, te_x_qry_self_attns = encoder(te_x_qry)
                for n in range(args.n_way):
                    te_x_spts = te_x_spt_enc[n]
                    te_y_spt = te_y_spts[n]
                    te_y_spt_mask = torch.as_tensor(te_y_spt, dtype=bool).to(device)
                    te_x_qrys = te_x_qry_enc[n]
                    te_y_qrys = te_y_qry[n]
                    te_y_qry_mask = torch.as_tensor(te_y_qrys, dtype=bool).to(device)
                    # accs = maml(x_spt, y_spt, x_qry, y_qry, y_qry_mask)
                    for k in range(args.k_spt):
                        te_x = te_x_spts[k]
                        te_y = te_y_spt[k]
                        te_y_mask = te_y_spt_mask[k]

                        accs, recall, f1, pres = maml.finetunning(te_x, te_y, te_y_mask, te_x_qrys, te_y_qrys,
                                                              te_y_qry_mask)
                    accs_all_test.append(accs)
                    pres_all.append(pres)
                    recall_all.append(recall)
                    f1_all.append(f1)

                # [b, update_step+1]
            test_accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
            test_recall = np.array(recall_all).mean(axis=0).astype(np.float16)
            test_f1 = np.array(f1_all).mean(axis=0).astype(np.float16)
            test_pres = np.array(pres_all).mean(axis=0).astype(np.float16)

            print('Query_Test acc:', test_accs)
            print('Query_Test recall:', test_recall)
            print('Query_Test f1:', test_f1)
            print('Query_Test pres:', test_pres)
            with open('log1.txt', 'a', encoding='utf-8') as f:
                f.write('test' + ' ' + '\t' + str(accs) + '\n')
            with open('test_log.txt', 'a', encoding='utf-8') as f:
                f.write('test' + ' ' + '\t' + str(test_accs) + '\n')
                f.close()




if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=200)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=5)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default= 0.002)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default= 1e-05)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=5)

    args = argparser.parse_args()

    main()
