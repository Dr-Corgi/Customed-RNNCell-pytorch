# -*- coding:utf-8 -*-
"""
Created on 18/11/29 下午10:21.

Author: Ruizhang1993 (zhang1rui4@foxmail.com)
"""
import math
import torch
import torch.nn as nn

class CustomedLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(CustomedLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        h, c = hidden

        # Linear mappings
        preact = self.i2h(x)+self.h2h(h)

        # activations
        gates = preact[:, : 3* self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size: 2* self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        c_t = torch.mul(c, f_t)
        c_t = c_t + torch.mul(i_t, g_t)
        h_t = torch.mul(o_t, c_t.tanh())

        return h_t, (h_t, c_t)

class CustomedLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers, bias=True):
        super(CustomedLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.n_layers = n_layers

        self.cells = nn.ModuleList(
            [CustomedLSTMCell(input_size, hidden_size, bias)])
        self.cells.extend([CustomedLSTMCell(input_size + hidden_size, hidden_size, bias) for _ in range(1, n_layers)])

    def forward(self, x, hidden):

        h, c = hidden

        hs, cs = [], []
        for i in range(self.n_layers):
            if i == 0:
                o_, (h_, c_) = self.cells[i](x, (h[i], c[i]))
            else:
                inp_ = torch.cat([x, o_], -1)
                o_, (h_, c_) = self.cells[i](inp_, (h[i], c[i]))
            hs.append(h_)
            cs.append(c_)

        o = o_
        h = torch.stack(hs)
        c = torch.stack(cs)

        return o, (h, c)