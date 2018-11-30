# -*- coding:utf-8 -*-
"""
Created on 18/11/29 下午9:38.

Author: Ruizhang1993 (zhang1rui4@foxmail.com)
"""
import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from cells import CustomedLSTMCell, CustomedLSTM


class AttentionSeq2Seq(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 enc_hidden_size,
                 dec_hidden_size,
                 attn_hidden_size,
                 enc_num_layers=4,
                 dec_num_layers=1,
                 enc_bidirectional=True,
                 enc_dropout=0.5,
                 dec_dropout=0.2):
        super(AttentionSeq2Seq, self).__init__()
        # Share
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)

        # Encoder
        self.enc_hidden_size = enc_hidden_size
        self.enc_num_layers = enc_num_layers
        self.enc_bidirectional = enc_bidirectional

        self.encoder = nn.LSTM(embedding_size, enc_hidden_size,
                               num_layers=enc_num_layers,
                               dropout=enc_dropout,
                               bidirectional=enc_bidirectional)

        # Attention
        self.attn_hidden_size = attn_hidden_size
        # self.attn_projection = nn.Linear(enc_hidden_size*2, attn_hidden_size*2)
        self.attn = nn.Linear(enc_hidden_size+dec_hidden_size, attn_hidden_size)
        self.attn_v = nn.Parameter(torch.rand(attn_hidden_size))
        stdv = 1. / math.sqrt(self.attn_v.size(0))
        self.attn_v.data.uniform_(-stdv, stdv)

        # Decoder
        self.dec_hidden_size = dec_hidden_size
        self.dec_num_layers = dec_num_layers
        if dec_num_layers > 1:
            self.decoder_init_projection = nn.ModuleList([nn.Linear(enc_hidden_size, dec_hidden_size) for _ in range(dec_num_layers)])
            self.decode_cell = CustomedLSTM(enc_hidden_size + embedding_size, dec_hidden_size, n_layers=dec_num_layers)
        else:
            self.decoder_init_projection = nn.Linear(enc_hidden_size, dec_hidden_size)
            self.decode_cell = CustomedLSTMCell(enc_hidden_size + embedding_size, dec_hidden_size)

        self.decoder_dropout = nn.Dropout(dec_dropout, inplace=True)
        self.output_projection = nn.Linear(dec_hidden_size, vocab_size)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        max_len = trg.size(0)

        if torch.cuda.is_available():
            outputs = Variable(torch.zeros(max_len, batch_size, self.vocab_size)).cuda()
        else:
            outputs = Variable(torch.zeros(max_len, batch_size, self.vocab_size))

        src_emb = self.word_embedding(src)
        enc_outputs, (enc_h, enc_c) = self.encoder(src_emb)

        if self.enc_bidirectional:
            enc_outputs = (enc_outputs[:, :, :self.enc_hidden_size] +
                           enc_outputs[:, :, self.enc_hidden_size:])

        if self.dec_num_layers > 1:
            hn = torch.stack([self.decoder_init_projection[i](enc_h[-1]) for i in range(self.dec_num_layers)])
            cn = torch.stack([self.decoder_init_projection[i](enc_c[-1]) for i in range(self.dec_num_layers)])
        else:
            hn = self.decoder_init_projection(enc_h[-1])
            cn = self.decoder_init_projection(enc_c[-1])

        input_ = Variable(trg.data[0, :])

        for t in range(1, max_len):
            inp_emb = self.word_embedding(input_)
            inp_emb = self.decoder_dropout(inp_emb)

            last_hidden = hn[-1] if self.dec_num_layers > 1 else hn

            # enc_outputs -> T x B x (d*l)
            # hn -> B x (d*l)
            time_step = enc_outputs.size(0)                         # time_step -> T
            attn_h = last_hidden.repeat(time_step, 1, 1).transpose(0, 1)     # attn_h -> B x T x (d*l)
            enc_o = enc_outputs.transpose(0, 1)                     # enc_o -> B x T x (d*l)
            attn_energies = self.score(attn_h, enc_o)               # attn_energies -> B x T
            attn_weights = F.softmax(attn_energies, dim=1).unsqueeze(1) # attn_weights -> B x 1 x T
            context = attn_weights.bmm(enc_outputs.transpose(0, 1)) # context -> B x 1 x (d*l)
            context = context.transpose(0, 1).squeeze(0)   # context -> B x (d*l)

            inp_emb = torch.cat([context, inp_emb], 1)

            dec_out, (hn, cn) = self.decode_cell(inp_emb, (hn, cn))
            dec_out = self.output_projection(dec_out)
            dec_out = F.log_softmax(dec_out, dim=1)

            outputs[t] = dec_out
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = dec_out.data.max(1)[1]
            if torch.cuda.is_available():
                input_ = Variable(trg.data[t] if is_teacher else top1).cuda()
            else:
                input_ = Variable(trg.data[t] if is_teacher else top1)

        return outputs

    def score(self, hidden, encoder_outputs):
        conc = torch.cat([hidden, encoder_outputs], 2)              # conc -> B x T x (2*d*l)
        # conc = self.attn_projection(conc)
        energy = self.attn(conc)                                    # energy -> B x T x (attn_hidden)
        energy = energy.transpose(1, 2)                             # energy -> B x a_h x T
        v = self.attn_v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)     # v -> 1 x B x a_h
        energy = torch.bmm(v, energy)                               # energy -> 1 x B x T
        return energy.squeeze(1)                                    # return -> B x T
