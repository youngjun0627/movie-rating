import torch
import torch.nn as nn
import torch.nn.functional as F
from .slowfast_ft import get_slowfast
from .only_audio import Only_Audio 
from .GMU import GatedMultimodalLayer


class Custom_Model(nn.Module):
    def __init__(self, classes_num, label_num, device):
        super(Custom_Model, self).__init__()        
        self.audio_model = Only_Audio(class_num = classes_num, label_num = label_num).to(device)
        self.video_model = get_slowfast(device)
        feature_dim1 = 1024
        feature_dim2 = 2048
        feature_dim = feature_dim1 + feature_dim2
        input_dim = 2048
        output_dim1 = 512
        output_dim2 = 64
        output_dim3 = 4
        self.dp = nn.Dropout(0.5)
        self.gmu = GatedMultimodalLayer(feature_dim1, feature_dim2, input_dim)
        self.first_fcs = nn.ModuleList([nn.Linear(input_dim, output_dim1, bias = True) for _ in range(label_num)])
        self.second_fcs = nn.ModuleList([nn.Linear(output_dim1, output_dim2, bias = True) for _ in range(label_num)])
        self.finally_fcs = nn.ModuleList([nn.Linear(output_dim2, classes_num, bias = True) for _ in range(label_num)])
        # genre fc, age fc #
        self.genrefc = nn.Linear(input_dim, 9)
        self.agefc = nn.Linear(input_dim, 4)

    def forward(self, video, audio):
        audio = self.audio_model(audio)
        video = self.video_model(video)
        features = self.gmu(audio, video)
        classes = []
        for fc1, fc2, fc3 in zip(self.first_fcs, self.second_fcs, self.finally_fcs):
            x = self.dp(F.relu(fc1(features)))
            x = self.dp(F.relu(fc2(x)))
            x = fc3(x)
            classes.append(x)
        classes = torch.stack(classes)
        genre = self.genrefc(features)
        age = self.agefc(features)
        return classes, genre, age

