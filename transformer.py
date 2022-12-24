# 数据构建
# 导入必备的工具包

# 数学计算工具包
import math
import os
import random

import torch
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import DataLoader

from load_data import word2id, make_data

# device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cuda"
root_path = './data/'
src_vocab = word2id()
src_vocab_size = len(src_vocab)

torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)



# ====================================================================================================
# Transformer模型

# ====================================================================================================
# Transformer模型

# Transformer Parameters
d_model = 512  # Embedding Size（token embedding和position编码的维度）
d_ff = 1024  # FeedForward dimension (FeedForward层隐藏神经元个数,两次线性层中的隐藏层 512->2048->512，线性层是用来做特征提取的），当然最后会再接一个projection层
d_k = d_v = 64  # dimension of K(=Q), V（Q和K的维度需要相同，这里为了方便让K=V）
n_layers = 6  # number of Encoder of Decoder Layer（Block的个数）
n_heads = 4  # number of heads in Multi-Head Attention（有几套头）


class PositionalEncoding(nn.Module):
    """位置编码器类的初始化函数, 共有三个参数, 分别是d_model: 词嵌入维度,
              dropout: 置0比率, 让部分神经元失效，max_len: 每个句子的最大长度"""

    def __init__(self, d_model, dropout=0.1, max_len=35):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 增加维度，在第一维（下标为0开始）上增加“1”


        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # print('sin部分==', torch.sin(position*div_term).shape) # sin部分== torch.Size([60, 256])
        # print('cos部分==', torch.cos(position*div_term).shape)# cos部分== torch.Size([60, 256])

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# 返回True和False组成的MASK
def get_attn_pad_mask(seq_q, seq_k):

    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()  # 维度
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k] 构成一个立方体(batch_size个这样的矩阵)


# 返回0和1组成的MASK
def get_attn_subsequence_mask(seq):
    """生成向后遮掩的掩码张量, 参数size是掩码张量最后两个维度的大小, 它的最后两维形成一个方阵"""
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequence_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


# ==========================================================================================
# 自注意力层
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        说明：在encoder-decoder的Attention层中len_q(q1,..qt)和len_k(k1,...km)可能不同
        """
        # torch.matmul是tensor的乘法  sqrt() 方法返回数字x的平方根。将Q和K的转置相乘
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        # mask矩阵填充scores（用-1e9填充scores中与attn_mask中值为1位置相对应的元素）
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        # scores : [batch_size, n_heads, len_q, len_k] * V: [batch_size, n_heads, len_v(=len_k), d_v]
        context = torch.matmul(attn, V)  # context: [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):


    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        # MultiHeadAttention(
        #   (W_Q): Linear(in_features=512, out_features=512, bias=False)
        #   (W_K): Linear(in_features=512, out_features=512, bias=False)
        #   (W_V): Linear(in_features=512, out_features=512, bias=False)
        # )

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        # B: batch_size, S:seq_len, D: dim
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, Head, W) -trans-> (B, Head, S, W)
        #           线性变换               拆成多头
        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k] # K和V的长度一定相同，维度可以不同
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)


        # attn_mask: [batch_size, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)

        # context: [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)

        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).to(device)(output + residual), attn



class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).to(device)(output + residual)  # [batch_size, seq_len, d_model]


# 使用EncoderLayer类实现编码器层
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        # 首先将self_attn和feed_forward传入其中
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """E
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]  mask矩阵(pad mask or sequence mask)
        """
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        # 第一个enc_inputs * W_Q = Q
        # 第二个enc_inputs * W_K = K
        # 第三个enc_inputs * W_V = V
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V（未线性变换前）
        enc_outputs = self.pos_ffn(enc_outputs)
        # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)  # token Embedding
        self.pos_emb = PositionalEncoding(d_model)  # Transformer中位置编码时固定的，不需要学习
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, src_len]
        """
        enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        # Encoder输入序列的pad mask矩阵
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs,
                                               enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class WeiYunet(Data.Dataset):
    def __init__(self, sup_root, qry_root, mode, batchsz, n_way, k_shot):  #
        """

        :param mode: train, val or test  模型：val是训练过程中的测试集
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        """
        self.batchsz = batchsz  # batch of set
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.mode = mode

        self.sup_src, self.sup_trg = make_data(root_path + sup_root + '.src', root_path + sup_root + '.trg')
        self.qry_src, self.qry_trg = make_data(root_path + qry_root + '.src', root_path + qry_root + '.trg')
        if self.mode =='train':
            self.cls_num = 200
        else:
            self.cls_num = 100
        self.create_batch(self.batchsz)

    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        ×episode× 表示批处理，表示我们要保留多少个集合。
        :param episodes: batch size
        :return:
        """
        # support_x = torch.FloatTensor(self.n_way, 1, 35)

        support_x_batch = []  # support set batch
        support_y_batch = []
        query_x_batch = []  # query set batch
        query_y_batch = []
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            selected_qur = np.random.choice(self.cls_num, 1, False)  # 无重复随机选
            # indexDtest = np.array(selected_qur)  # idx for Dtest
            query_x, query_y = [], []
            for id in selected_qur:
                Dtest_x = torch.tensor(self.qry_src[id])
                Dtest_y = torch.tensor(self.qry_trg[id])
                query_x.append(Dtest_x)  # 获取当前 Dtrain 的所有数据
                query_y.append(Dtest_y)

            query_xs = torch.stack(query_x, 0)
            query_ys = torch.stack(query_y, 0)
            # np.random.shuffle(selected_qur)  #随机打乱数据
            support_x, support_y = [], []
            for cls in selected_qur:
                # 2. select k_shot + k_query for each class
                # if self.mode == 'train':
                selected_idx = random.sample(range((cls-1)*10,cls*10), self.k_shot)
                for id in selected_idx:
                    Dtrain_x = torch.tensor(self.sup_src[id])
                    Dtrain_y = torch.tensor(self.sup_trg[id])
                    support_x.append(Dtrain_x)  # 获取当前 Dtrain 的所有数据
                    support_y.append(Dtrain_y)

                support_xs = torch.stack(support_x, 0)
                support_ys = torch.stack(support_y, 0)

            support_x_batch.append(support_xs)  # append set to current sets
            self.support_x_batch = torch.stack(support_x_batch, 0)
            support_y_batch.append(support_ys)
            self.support_y_batch = torch.stack(support_y_batch, 0)
            query_x_batch.append(query_xs)  # append sets to current sets
            self.query_x_batch = torch.stack(query_x_batch, 0)
            query_y_batch.append(query_ys)
            self.query_y_batch = torch.stack(query_y_batch, 0)


    def __getitem__(self, idx):
        return self.support_x_batch[idx],self.support_y_batch[idx],self.query_x_batch[idx],self.query_y_batch[idx]

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz


print("transformer ok")
