import torch
import torch.nn as nn
from .slowfast_ft import get_slowfast
from .only_audio import Only_Audio 

class Custom_Model(nn.Module):
    def __init__(self, classes_num, label_num, device):
        super(Custom_Model, self).__init__()        
        self.audio_model = Only_Audio(class_num = classes_num, label_num = label_num).to(device)
        self.video_model = get_slowfast(device)
        feature_dim = 2048+1024
        input_dim = 2048
        self.compress_fc = nn.Conv2d(feature_dim, input_dim, kernel_size = 1, stride = 1)
        self.dp = nn.Dropout(0.5)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.fcs = nn.ModuleList([nn.Linear(input_dim, classes_num) for _ in range(label_num)])
        # genre fc, age fc #
        self.genrefc = nn.Linear(input_dim, 9)
        self.agefc = nn.Linear(input_dim, 4)

    def forward(self, video, audio):
        audio = self.audio_model(audio)
        video = self.video_model(video)
        features = torch.cat([audio,video], axis = 1).unsqueeze(2).unsqueeze(3)
        features = self.dp(self.leaky_relu(self.compress_fc(features))).squeeze(3).squeeze(2)
        classes = []
        for fc in self.fcs:
            classes.append(fc(features))
        classes = torch.stack(classes)
        genre = self.genrefc(features)
        age = self.agefc(features)
        return classes, genre, age

