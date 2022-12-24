import torch
from sklearn.metrics import precision_score,recall_score,f1_score
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import optim
import numpy as np

from learner import Learner
from copy import deepcopy

import warnings
warnings.filterwarnings("ignore")

# def adjust_learning_rate(epoch, lr):
#     """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
#     if epoch // 10000:
#         lr *= (0.1 ** (epoch // 2))


class Meta(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.net = Learner(config)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.梯度归一化，解决梯度爆炸
        :param grad: list of gradients
        :param max_norm: maximum norm allowable  最大允许范围
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)  # 其实就是对g张量L2范数，先对y中每一项取平方，之后累加，最后取根号
            total_norm += param_norm.item() ** 2  # 用item得到元素值平方累加
            counter += 1
        total_norm = total_norm ** (1. / 2)  # 开方

        clip_coef = max_norm / (total_norm + 1e-6)  # 剪辑系数
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm / counter

    def forward(self, x_spt, y_spt, x_qry, y_qry, qry_mask):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        # task_num, setsz, c_, h, w = x_spt.size
        task_num, h, w = x_spt.size()
        # y_spt = y_spt.resize(task_num,setsz*35)
        # print("task_num",x_spt.size())
        # y_spt =y_spt.view(y_spt.size(0), -1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]
        num = 0
        losses = []

        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True)  # 32*1  1*32

            qry_masks = qry_mask[i].squeeze()

            num_not_pad_tokens = (qry_masks == 1).sum().tolist()
            num += num_not_pad_tokens
            qry_y = torch.masked_select(y_qry[i], qry_masks).cuda()

            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())

            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)[0:num_not_pad_tokens]
                loss_q = F.cross_entropy(logits_q[0:num_not_pad_tokens], qry_y.view(-1))
                losses.append(loss_q)
                losses_q[0] += loss_q

                pred_q = F.relu(logits_q).argmax(dim=1)
                # pred_q = torch.masked_select(pred_q, masks).cuda()

                correct = torch.eq(pred_q, qry_y).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q[0:num_not_pad_tokens], qry_y.view(-1))
                losses.append(loss_q)
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.relu(logits_q).argmax(dim=1)

                pred_q = torch.masked_select(pred_q, qry_masks).cuda()

                correct = torch.eq(pred_q, qry_y).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                if k < self.update_step - 1:
                    with torch.no_grad():
                        logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                        loss_q = F.cross_entropy(logits_q[0:num_not_pad_tokens], qry_y)
                        losses.append(loss_q)
                        losses_q[k + 1] += loss_q
                else:
                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                    loss_q = F.cross_entropy(logits_q[0:num_not_pad_tokens], qry_y)
                    losses.append(loss_q)
                    losses_q[k + 1] += loss_q
                #
                with torch.no_grad():
                    pred_q = F.relu(logits_q[0:num_not_pad_tokens]).argmax(dim=1)

                    # pred_q = torch.masked_select(pred_q, masks).cuda()

                    correct = torch.eq(pred_q, qry_y).sum().item()
                    corrects[k + 1] = corrects[k + 1] + correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num  # -  0.01 * max(losses)

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward(retain_graph=True)
        self.meta_optim.step()

        accs = np.array(corrects) / np.array(num)  # 50*

        return accs




    def finetunning(self, x_spt, y_spt, spt_mask, x_qry, y_qry, qry_mask):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 2

        # for n, x in enumerate(masks):
        #     m = list(map(lambda y: 0 if y else 1, x))
        #     masks = torch.BoolTensor(m).cuda()
        # mask = torch.ByteTensor(y_qry[i] > 0)
        num_not_pad_token = (spt_mask == 1).sum().tolist()
        y_s = torch.masked_select(y_spt, spt_mask).cuda()

        num_not_pad_tokens = (qry_mask == 1).sum().tolist()
        y_q = torch.masked_select(y_qry, qry_mask).cuda()
        # print('y',y_q)
        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)[:num_not_pad_token]
        loss = F.cross_entropy(logits, y_s)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))
        logits_q = net(x_qry, net.parameters(), bn_training=True)[0:num_not_pad_tokens]
        # [setsz]
        pred_q = F.relu(logits_q).argmax(dim=1)
        # print('0',pred_q)


        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)[0:num_not_pad_tokens]
            # [setsz]
            pred_q = F.relu(logits_q).argmax(dim=1)


            # scalar

            # pred_q = torch.masked_select(pred_q, mask).cuda()

            correct = torch.eq(pred_q, y_q).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]

            pred_q = F.relu(logits_q).argmax(dim=1)
            # scalar

            pred_q = torch.masked_select(pred_q, qry_mask).cuda()

            correct = torch.eq(pred_q, y_q).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)[0:num_not_pad_token]
            loss = F.cross_entropy(logits, y_s)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)[0:num_not_pad_tokens]
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_q)

            with torch.no_grad():
                pred_q = F.relu(logits_q).argmax(dim=1)

                # pred_q = torch.masked_select(pred_q, mask).cuda()

                correct = torch.eq(pred_q, y_q).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct
        # print(pred_q)
        # accs , recall ,f1 = evaluate(y_q,pred_q)
        del net

        accs = np.array(corrects) / np.array(num_not_pad_tokens)
        pres = precision_score(np.array(y_q.cpu()),np.array(pred_q.cpu()),average='weighted')
        recall =recall_score(np.array(y_q.cpu()),np.array(pred_q.cpu()), average='weighted')
        # recalls = []
        # for r in recall:
        #     if r==None:
        #         r=0
        #     recalls.append(r)
        # print(recall)
        f1 =f1_score(y_true = np.array(y_q.cpu()),
                    y_pred = np.array(pred_q.cpu()),
                     average="weighted")
        # f1s=[]
        # for f in f1:
        #     if f==None:
        #         f=0
        #     f1s.append(f)
        # print(accs)

        return accs,recall,f1 ,pres


def main():
    pass


if __name__ == '__main__':
    main()
