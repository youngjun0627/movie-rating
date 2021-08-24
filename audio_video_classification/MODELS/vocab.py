from collections import Counter

import torchwordemb
import torch

import util as ut

class GloveVocabBuilder(object) :

    def __init__(self, path_glove):
        self.vec = None
        self.vocab = None
        self.path_glove = path_glove

    def get_word_index(self, padding_marker='__PADDING__', unknown_marker='__UNK__',):
        _vocab, _vec = torchwordemb.load_glove_text(self.path_glove)
        vocab = {padding_marker:0, unknown_marker:1}
        for tkn, indx in _vocab.items():
            vocab[tkn] = indx + 2
        vec_2 = torch.zeros((2, _vec.size(1)))
        vec_2[1].normal_()
        self.vec = torch.cat((vec_2, _vec))
        self.vocab = vocab
        return self.vocab, self.vec

