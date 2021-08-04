import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Only_Text(nn.Module):
    def __init__(self, class_num=4, label_num = 5, dropout=0.7, mode='multi', vocab_size = 0):
        super(Only_Text, self).__init__()
        self.mode = mode
        self.embed_dim = 80
        self.class_num = class_num
        self.label_num = label_num
        
        
        self.embedding = None
        self.textdp = nn.Dropout(0.5)
        self.textcnn = nn.Sequential(
                    nn.Conv1d(in_channels = 60, out_channels = self.embed_dim, kernel_size = 7, padding = 3),
                    nn.ReLU(inplace = True),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Conv1d(in_channels = self.embed_dim, out_channels = self.embed_dim, kernel_size = 7, padding = 3), 
                    nn.ReLU(inplace = True),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Conv1d(in_channels = self.embed_dim, out_channels = self.embed_dim*2, kernel_size = 3, padding = 1), 
                    nn.ReLU(inplace = True),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Conv1d(in_channels = self.embed_dim*2, out_channels = self.embed_dim*2, kernel_size = 3, padding = 1), 
                    nn.ReLU(inplace = True),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Conv1d(in_channels = self.embed_dim*2, out_channels = self.embed_dim*2, kernel_size = 3, padding = 1), 
                    nn.AdaptiveAvgPool1d(1)
                    ) 

        self.classifier1 = nn.ModuleList([nn.Linear(self.embed_dim*2, class_num) for _ in range(label_num)])
        # genre fc, age fc #
        self.genrefc = nn.Linear(self.embed_dim*2, 9)
        self.agefc = nn.Linear(self.embed_dim*2, 4)

        #self.lstm = nn.LSTM(self.fast_inplanes + 2048, hidden_size = 32, num_layers = 2, batch_first=False, bidirectional=True, dropout=0.4) 
        
        
    def init_text_weights(self, vocab_size):
        self.embedding = nn.Embedding(vocab_size, self.embed_dim, padding_idx=0)
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        for m in self.textcnn:
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

    def forward(self, text):

        text = self.embedding(text)
        text = self.textdp(text)
        text = self.textcnn(text)
        features = self.textdp(text.view(text.size(0), -1))
        
        outputs = []
        for fc in self.classifier1:
            outputs.append(fc(features))
        outputs = torch.stack(outputs)
        genre = self.genrefc(features)
        age = self.agefc(features)
        return outputs, genre, age
    



if __name__=='__main__':
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    num_classes = 4
    num_label = 5
    mode = 'multi'
    model = Only_Text()

