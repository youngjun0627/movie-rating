import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
import numpy as np

class TextModel(nn.Module):
    def __init__(self, vocab_size):
        super(TextModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, 256, sparse = True)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128,64)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.bias.data.zero_()

    def forward(self, text, offsets):
        x = self.embedding(text,offsets)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

if __name__=='__main__':
    csv_file = '../UTILS/train-for_user.csv'
    tokenizer = get_tokenizer('basic_english')
    counter = Counter()
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        rdr = csv.reader(f)
        for data in rdr:
            plot = data[7]
            counter.update(tokenizer(plot))
        vocab = Vocab(counter, min_freq=1)
        text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
        f.close()

    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        rdr = csv.reader(f)
        for data in rdr:
            plot = data[7]
            print(plot)
            processed_text = np.array(text_pipeline(plot), dtype = np.long)
            print(processed_text)
            offset = processed_text.shape[0]
            
            print(offset)
        print(len(vocab))
