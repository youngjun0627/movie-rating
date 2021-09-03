import torch
import torch.nn as nn
import torch.nn.functional as F

'''
reference code: https://github.com/IsaacRodgz/GMU-Baseline/blob/master/src/models.py
'''

class GatedMultimodalLayer(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim):
        super(GatedMultimodalLayer, self).__init__()
        self.hidden1 = nn.Linear(input_dim1, output_dim, bias = False)
        self.hidden2 = nn.Linear(input_dim2, output_dim, bias = False)
        self.hidden3 = nn.Linear(output_dim * 2, 1, bias = False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.tanh(self.hidden1(x1))
        h2 = self.tanh(self.hidden2(x2))
        x = torch.cat([h1,h2], dim=1)
        z = self.sigmoid(self.hidden3(x))
        return z.view(z.size(0), 1) * h1 + (1-z).view(z.size(0), 1) * h2

if __name__ == '__main__':
    feature_size1 = 512
    feature_size2 = 2048
    a = torch.randn(size = (2, feature_size1))
    b = torch.randn(size = (2, feature_size2))
    model = GatedMultimodalLayer(feature_size1, feature_size2, feature_size2)
    o = model(a,b)
    print(o.shape)
