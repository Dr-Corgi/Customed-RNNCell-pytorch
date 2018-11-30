# -*- coding:utf-8 -*-
"""
Created on 18/8/1 下午5:07.

Author: Ruizhang1993 (zhang1rui4@foxmail.com)
"""
import re
import abc
import json
import codecs


class Corpus(object):

    def __init__(self, corpus_path, max_src_vocab=None, max_trg_vocab=None, word2vec_path=None, add_start_token=True):

        self.corpus_path = corpus_path
        self.word2vec_path = word2vec_path
        self.word2vec = None

        self.max_src_vocab = max_src_vocab
        self.max_trg_vocab = max_trg_vocab
        self.load_data(add_start_token)


    @abc.abstractmethod
    def load_data(self, add_start_token):
        raise NotImplementedError()

    @abc.abstractmethod
    def to_id_corpus(self, data):
        raise NotImplementedError()


class NlpccCorpus(Corpus):

    def load_data(self, add_start_token):

        src_words = {}
        trg_words = {}

        special_token = ['<pad>', '<unk>', '<s>', '</s>']

        self.pad_id = 0
        self.unk_id = 1
        self.go_id = 2
        self.eos_id = 3

        data = []

        raw_data = json.load(codecs.open(self.corpus_path, 'r', 'utf8'))

        for (src, _), (trg, _) in raw_data:
            if add_start_token:
                new_src = ["<s>"] + src.strip().split(" ") + ["</s>"]
                new_trg = ["<s>"] + trg.strip().split(" ") + ["</s>"]
            else:
                new_src = src.strip().split(" ") + ["</s>"]
                new_trg = trg.strip().split(" ") + ["</s>"]

            data.append((new_src, new_trg))

            for word in new_src:
                if word not in special_token:
                    src_words[word] = src_words.get(word, 0) + 1
            for word in new_trg:
                if word not in special_token:
                    trg_words[word] = trg_words.get(word, 0) + 1

        del raw_data

        # create vocab
        sorted_src_words = special_token + [w for w, _ in sorted(src_words.items(), key=lambda x: x[1], reverse=True)]
        if self.max_src_vocab:
            self.word2idx = sorted_src_words[:self.max_src_vocab]
        else:
            self.word2idx = sorted_src_words
        self.idx2word = {t: idx for idx, t in enumerate(self.word2idx)}

        self.data = data

    def to_id_corpus(self, data):
        results = []

        for src, trg in data:
            id_src = [self.idx2word.get(t, self.unk_id) for t in src]
            id_trg = [self.idx2word.get(t, self.unk_id) for t in trg]

            results.append((id_src, id_trg))

        return results