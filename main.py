# -*- coding:utf-8 -*-
"""
Created on 18/8/1 下午3:59.

Author: Ruizhang1993 (zhang1rui4@foxmail.com)
"""
import argparse
import math
import numpy as np
import os

import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable

from data_utils.corpus import NlpccCorpus
from data_utils.data_loader import DataLoader
# from model import Seq2Seq, Encoder, Decoder, PureDecoder
from normal_seq2seq import Seq2Seq, AttentionSeq2Seq

def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=20,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=64,
                   help='number of batch_size for train')
    p.add_argument('-lr', type=int, default=0.0001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=5.0,
                   help='grad clip')

    p.add_argument('-hidden_size', type=int, default=128,
                   help='hidden size')
    p.add_argument('-embedding_size', type=int, default=128,
                   help='embedding size')

    return p.parse_args()

def evaluate(model, data_loader, trg_vocab_size, batch_size, pad_id=0):
    model.eval()
    total_loss = 0

    data_loader.epoch_init(batch_size, shuffle=False)
    for i in range(data_loader.num_batch):
        src, src_lens, trg, trg_lens = data_loader.next_batch()
        with torch.no_grad():
            if torch.cuda.is_available():
                src = Variable(src.cuda())
                trg = Variable(trg.cuda())
            else:
                src = Variable(src)
                trg = Variable(trg)
        output = model(src, trg)
        loss = F.cross_entropy(output[1:].view(-1, trg_vocab_size),
                               trg[1:].contiguous().view(-1),
                               ignore_index=pad_id)
        total_loss += loss.data[0]

        del loss
        del src
        del trg

    return total_loss / data_loader.num_batch

def train(model, optimizer, data_loader, batch_size, trg_vocab_size, grad_clip, pad_id=0):
    model.train()
    total_loss = 0

    data_loader.epoch_init(batch_size)
    for i in range(data_loader.num_batch):
        optimizer.zero_grad()

        src, src_lens, trg, trg_lens = data_loader.next_batch()

        with torch.no_grad():
            if torch.cuda.is_available():
                src = Variable(src.cuda())
                trg = Variable(trg.cuda())
            else:
                src = Variable(src)
                trg = Variable(trg)

        output = model(src, trg)
        loss = F.cross_entropy(output[1:].view(-1, trg_vocab_size),
                               trg[1:].contiguous().view(-1),
                               ignore_index=pad_id)
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()

        if i % 100 == 0 and i != 0:
            total_loss = total_loss / 100
            print("[%d][loss:%5.2f][pp:%5.2f]" %
                  (i, total_loss, math.exp(total_loss)))
            total_loss = 0

        del loss
        del src
        del trg

def main():

    args = parse_arguments()

    print('Preparing dataset...')
    # corpus = FraCorpus("./datasets/fra.txt")
    corpus = NlpccCorpus("./datasets/train_data.json",
                         max_src_vocab=30000,
                         max_trg_vocab=30000)

    data = corpus.to_id_corpus(corpus.data)

    np.random.shuffle(data)

    train_data = DataLoader("Train", data[:-2048])    # train_data ==> data[:-1000]
    val_data =  DataLoader("Evaluate", data[-2048:-1024])
    test_data = DataLoader("Test", data[-1024:])  # train_data ==> data[:-1000]

    print('Initialling models...')

    src_vocab_size = trg_vocab_size = len(corpus.word2idx)

    # encoder = Encoder(src_vocab_size,
    #                   args.embedding_size,
    #                   args.hidden_size,
    #                   n_layers=2,
    #                   dropout=0.5)
    #
    # decoder = PureDecoder(args.embedding_size,
    #                   args.hidden_size,
    #                   trg_vocab_size,
    #                   n_layers=1,
    #                   dropout=0.5)
    # if torch.cuda.is_available():
    #     seq2seq = Seq2Seq(encoder, decoder).cuda()
    # else:
    #     seq2seq = Seq2Seq(encoder, decoder)

    if torch.cuda.is_available():
        seq2seq = AttentionSeq2Seq(src_vocab_size,
                          args.embedding_size,
                          args.hidden_size*2,
                          args.hidden_size,
                          args.hidden_size//2).cuda()
    else:
        seq2seq = AttentionSeq2Seq(src_vocab_size,
                          args.embedding_size,
                          args.hidden_size*2,
                          args.hidden_size,
                          args.hidden_size//2).cuda()

    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    print(seq2seq)

    best_val_loss = None
    for e in range(args.epochs):
        train(seq2seq, optimizer, train_data, args.batch_size, trg_vocab_size, args.grad_clip)
        val_loss = evaluate(seq2seq, val_data, trg_vocab_size, batch_size=32)
        print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS"
              % (e, val_loss, math.exp(val_loss)))

        if not best_val_loss or val_loss < best_val_loss:
            print("[!] saving model...")
            if not os.path.isdir(".save"):
                os.makedirs(".save")
            torch.save(seq2seq.state_dict(), './.save/seq2seq_%d.pt' % (e))
            best_val_loss = val_loss

    test_loss = evaluate(seq2seq, test_data, trg_vocab_size, batch_size=1)
    print("[TEST] loss:%5.2f" % test_loss)


if __name__ == '__main__':
    main()
