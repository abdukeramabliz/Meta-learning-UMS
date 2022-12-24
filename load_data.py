# -*- coding: utf-8 -*-
import heapq
# import json
import linecache #读取索引行的内容
# 读取数据集
import torch


def read_data(file_path):
    # 读取数据集
    index = [-1]
    with open(file_path, "r", encoding="utf-8") as f:
        content = [_.strip() for _ in f.readlines()]  # 每行进行切分
        index.append(len(content))  # 行数

    # 读取空行所在的行号
    # index.extend([i for i, _ in enumerate(content) if ' ' not in _])
    # index.append(len(content))

    sentences = []
    for j in range(len(index) - 1):
        sents, tags = [], []
        segment = content[index[j] + 1: index[j + 1]]
        pad_size = 35
        for line in segment:
            tag = line.split()
            if len(tag) >= pad_size:
                pad_size = len(tag)
        for line in segment:
            tag = line.split()

            if len(tag) >= pad_size:
                pad_size = len(tag)
                sents.append(line)
                tags.append(line.split())
            else:
                # mask为1的是实际数据，0是填充补位，无数据不考虑
                line_split = line.split() + ["0"] * (pad_size - len(tag))
                tags.append(line_split)
                sents.append(' '.join(line_split))

        # sentences.append(''.join(sent))
        # tags.append(tag)

    return sents, tags


def read_u_data(file_path):
    # 读取数据集
    index = [-1]
    with open(file_path, "r", encoding="utf-8") as f:
        content = [_.strip() for _ in f.readlines()]  # 每行进行切分
        index.append(len(content))  # 行数

    # 读取空行所在的行号
    # index.extend([i for i, _ in enumerate(content) if ' ' not in _])
    # index.append(len(content))

    sentences = []
    for j in range(len(index) - 1):
        sents, tags = [], []
        segment = content[index[j] + 1: index[j + 1]]
        pad_size = 35
        for line in segment:
            tag = line.split()
            line_split = line.split()
            lines =  ["0"] * (pad_size - len(tag))+line_split[::-1]
            tags.append(lines)
            sents.append(' '.join(lines))
            # print(tags)

        # sentences.append(''.join(sent))
        # tags.append(tag)

    return sents,tags


# 读取训练集数据
# 将标签转换成id
def label2id():
    label_id_dict = {"0": 0, "b": 1, "m": 2, "e": 3, "s": 4}
    return label_id_dict


def word2id():
    word_id_dict = {
        "0": 0, "ب": 1, "ې": 2, "ك": 3, "ى": 4, "ت": 5, "ە": 6, "ل": 7, "م": 8, "ي"
        : 9, "د": 10, "غ": 11, "ا": 12, "ن": 13, "ق": 14, "ئ": 15, "ش": 16, "ز": 17,
        "و": 18, "ڭ": 19, "گ": 20, "ۆ": 21, "ر": 22, "ۈ": 23, "س": 24, "ۋ": 25, "خ": 26,
        "ف": 27, "ج": 28, "ۇ": 29, "چ": 30, "پ": 31, "ھ": 32, "،": 33, "ژ": 34
    }

    return word_id_dict


# 数据构建
def make_data(sent_path, trg_path):
    """把单词序列转换为数字序列"""
    sent, _ = read_data(sent_path)
    trg, _ = read_data(trg_path)
    sentences = []
    for i in range(len(sent)):
        sentence = []
        sentence.append(sent[i])
        sentence.append(trg[i])
        # sentence.append(trg[i])
        sentences.append(sentence)
    enc_inputs, dec_inputs = [], []
    src_vocab = word2id()
    # src_idx2word = {i: w for i, w in enumerate(src_vocab)}
    # src_vocab_size = len(src_vocab)
    tgt_vocab = label2id()
    # idx2word = {i: w for i, w in enumerate(tgt_vocab)}
    # tgt_vocab_size = len(tgt_vocab)

    # src_len = len(src_vocab) - 1  # （源句子的长度）enc_input max sequence length
    # tgt_len = len(tgt_vocab) - 1  # dec_input(=dec_output) max sequence length

    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]  # [[1,2,3,4,5,6,7,0]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]  # [[9,1,2,3,4,5,6,7,11]]
        # dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]  # [[1,2,3,4,5,6,7,11,10]]

        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        # dec_outputs.extend(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(
        dec_inputs)  # ,torch.BoolTensor(dec_inputs) # , torch.LongTensor(dec_outputs)


def get_line_context(file_path, line_number):
    return linecache.getline(file_path, line_number+1).strip()

def similarity(tes_inputs, enc_inputs):
    sims_k_ids = []
    for i in tes_inputs:
        sim = []
        for j in enc_inputs:
            j = j.float()
            s = torch.cosine_similarity(i, j, dim=0)
            sim.append(s)

        sim_k = heapq.nlargest(10, sim)
        sim_k_ids=[]
        for k in sim_k:
            sim_k_id=sim.index(k,0,len(sim))
            sim_k_ids.append(sim_k_id)  #k个最高相似度索引
        # print(sim_k_ids)
        sims_k_ids.append(torch.tensor(sim_k_ids))
    return sims_k_ids
    #

if __name__ == '__main__':
    enc_inputs, dec_inputs = make_data("./data/meta_test.src", "./data/meta_test.trg")

    tes_inputs, tes_outputs = make_data("data/meta_train_query.src", "data/meta_train_query.trg")

    s = similarity(tes_inputs, enc_inputs)

    for n in range(len(s)):
        for m in range(10):
            #print(s[n][m])
            line=get_line_context('./data/meta_train.src',s[n][m])
            with open('data/meta_train_support.src', 'a', encoding='utf=8')as f:
                f.write(line+'\n')

    for n in range(len(s)):
        for m in range(10):
            #print(s[n][m])
            line=get_line_context('./data/meta_train.trg',s[n][m])
            with open('data/meta_train_support.trg', 'a', encoding='utf=8')as f:
                f.write(line+'\n')
    #




