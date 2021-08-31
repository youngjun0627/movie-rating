import torch
import torch.nn as nn
from .slowfast_ft import get_slowfast
from .only_text2 import LSTM_with_Attention

class Custom_Model(nn.Module):
    def __init__(self, classes_num, label_num, device):
        super(Custom_Model, self).__init__()        
        self.text_model = LSTM_with_Attention(class_num = classes_num, label_num = label_num).to(device)
        self.video_model = get_slowfast(device)
        feature_dim = 2048+512
        input_dim = 2048
        self.compress_fc = nn.Conv2d(feature_dim, input_dim, kernel_size = 1, stride = 1)
        self.dp = nn.Dropout(0.5)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.fcs = nn.ModuleList([nn.Linear(input_dim, classes_num) for _ in range(label_num)])
        # genre fc, age fc #
        self.genrefc = nn.Linear(input_dim, 9)
        self.agefc = nn.Linear(input_dim, 4)

    def forward(self, video, text):
        text = self.text_model(text)
        video = self.video_model(video)
        features = torch.cat([text,video], axis = 1).unsqueeze(2).unsqueeze(3)
        features = self.dp(self.leaky_relu(self.compress_fc(features))).squeeze(3).squeeze(2)
        classes = []
        for fc in self.fcs:
            classes.append(fc(features))
        classes = torch.stack(classes)
        genre = self.genrefc(features)
        age = self.agefc(features)
        return classes, genre, age

if __name__=='__main__':
    device = torch.device('cuda:1')
    video = torch.rand(size=(2,3,1024,112,112)).to(device)
    text = torch.randint(0,1000,size=(2,300)).to(device)
    model = Custom_Model(4,4,device).to(device)
    model(video, text)
    video = torch.rand(size=(1,3,1024,112,112)).to(device)
    text = torch.randint(0,1000,size=(1,300)).to(device)
    model = Custom_Model(4,4,device).to(device)
    model(video, text)
