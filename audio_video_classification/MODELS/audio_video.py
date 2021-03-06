import torch
import torch.nn as nn
from .slowfast_ft import get_slowfast
from .only_audio import Only_Audio 

class Custom_Model(nn.Module):
    def __init__(self, classes_num, label_num, device):
        super(Custom_Model, self).__init__()        
        self.audio_model = Only_Audio(class_num = classes_num, label_num = label_num).to(device)
        self.video_model = get_slowfast(device)
        input_dim = 2048+1024
        self.fcs = nn.ModuleList([nn.Linear(input_dim, classes_num) for _ in range(label_num)])
        # genre fc, age fc #
        self.genrefc = nn.Linear(input_dim, 9)
        self.agefc = nn.Linear(input_dim, 4)

    def forward(self, video, audio):
        audio = self.audio_model(audio)
        video = self.video_model(video)
        features = torch.cat([audio,video], axis = 1)

        classes = []
        for fc in self.fcs:
            classes.append(fc(features))
        classes = torch.stack(classes)
        genre = self.genrefc(features)
        age = self.agefc(features)
        return classes, genre, age

