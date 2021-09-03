import torch
import torch.nn as nn
from .slowfast_ft import get_slowfast
from .only_text2 import LSTM_with_Attention
from .GMU import GatedMultimodalLayer

class Custom_Model(nn.Module):
    def __init__(self, classes_num, label_num, device):
        super(Custom_Model, self).__init__()        
        self.text_model = LSTM_with_Attention(class_num = classes_num, label_num = label_num).to(device)
        self.video_model = get_slowfast(device)
        feature_dim1 = 512
        feature_dim2 = 2048
        feature_dim = feature_dim1 + feature_dim2
        input_dim = 2048
        self.gmu = GatedMultimodalLayer(feature_dim1, feature_dim2, input_dim)
        self.fcs = nn.ModuleList([nn.Linear(input_dim, classes_num) for _ in range(label_num)])
        # genre fc, age fc #
        self.genrefc = nn.Linear(input_dim, 9)
        self.agefc = nn.Linear(input_dim, 4)

    def forward(self, video, text):
        text = self.text_model(text)
        video = self.video_model(video)
        features = self.gmu(text, video)
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
