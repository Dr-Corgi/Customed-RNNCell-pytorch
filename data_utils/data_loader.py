# -*- coding:utf-8 -*-
"""
Created on 18/8/1 下午5:13.

Author: Ruizhang1993 (zhang1rui4@foxmail.com)
"""
import numpy as np
import torch

class DataLoader(object):

    def __init__(self, name, data):

        self.name = name
        self.data = data
        self.data_size = len(data)

        self.src_lens = all_lens = [len(src) for src, _ in self.data]
        self.trg_lens = [len(trg) for _, trg in self.data]

        self.indexes = list(np.argsort(all_lens))[::-1]

        self.num_batch = None

    def pad_to(self, tokens, max_utt_len, do_pad=True):
        if len(tokens) >= max_utt_len:
            return tokens[0:max_utt_len-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (max_utt_len - len(tokens))
        else:
            return tokens

    def _shuffle_batch_indexes(self):
        np.random.shuffle(self.batch_indexes)

    def epoch_init(self, batch_size, shuffle=True):
        assert len(self.indexes) == self.data_size and len(self.src_lens) == self.data_size

        self.ptr = 0
        self.batch_size = batch_size

        # create batch indexes
        temp_num_batch = self.data_size // batch_size
        self.batch_indexes = []

        for i in range(temp_num_batch):
            temp_batch_indexes = self.indexes[i*batch_size:(i+1)*batch_size]
            if shuffle:
                np.random.shuffle(temp_batch_indexes)
            self.batch_indexes.append(temp_batch_indexes)

        left_over = self.data_size - temp_num_batch * batch_size

        if shuffle:
            self._shuffle_batch_indexes()

        self.num_batch = len(self.batch_indexes)
        print("%s begins with %d batches with %d left for samples" % (self.name, self.num_batch, left_over))

    def next_batch(self):
        if self.ptr < self.num_batch:
            current_batch = self.batch_indexes[self.ptr]
            self.ptr += 1
            return self._prepare_batch(current_batch=current_batch)
        else:
            return None

    def _prepare_batch(self, current_batch):

        batch_data = [self.data[idx] for idx in current_batch]

        src, src_lens, trg, trg_lens = [], [], [], []

        max_src_lens = max([len(t_) for t_, _ in batch_data])
        max_trg_lens = max([len(t_) for _, t_ in batch_data])

        for s, t in batch_data:
            s_paded = self.pad_to(s, max_src_lens)
            src.append(s_paded)
            src_lens.append(len(s))
            trg.append(self.pad_to(t, max_trg_lens))
            trg_lens.append(len(t))

        src = torch.from_numpy(np.array(src)).transpose(0, 1)
        src_lens = torch.from_numpy(np.array(src_lens))
        trg = torch.from_numpy(np.array(trg)).transpose(0, 1)
        trg_lens = torch.from_numpy(np.array(trg_lens))

        return src, src_lens, trg, trg_lens
