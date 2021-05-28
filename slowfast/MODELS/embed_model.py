import torch
import torch.nn as nn
import torch.nn.Functional as F


class Embed(nn.Module):
    def __init__(self, vocap_size, vector_size):
        super(Embed, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vector_size)
        self.conv1 = nn.Conv1d(vector_size,32,8)
        self.maxpool = nn.MaxPool1d(2)
        self.relu = nn.ReLU(inplace=True)
        #self.fc1 = nn.Linear(32,10)
        #self.fc2 = nn.Linear(10,1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x



