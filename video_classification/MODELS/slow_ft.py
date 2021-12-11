'''
Here's the snippet from my code.
Also modify the .yaml config file as needed.
Hope this helps! 
'''
import torch
import torch.nn as nn
import numpy as np
from .slowfast.models.build import build_model
from .slowfast.models import head_helper
from .slowfast.config.defaults import get_cfg
from .slowfast.models import checkpoint as cu 
import os

def get_facebook_model(device):
    # slowfast net
    cfg = get_cfg()
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Load config from cfg.
    root = '/home/uchanlee/uchanlee/movie-rating/video_classification/MODELS/slowfast'
    cfg_file = os.path.join(root,'config/slow_config.yaml')
    cfg.merge_from_file(cfg_file)
    model = build_model(cfg, gpu_id=device)
    cfg.TRAIN.CHECKPOINT_TYPE = 'caffe2'
    cfg.TRAIN.CHECKPOINT_FILE_PATH = os.path.join(root, 'saved_model/C2D_8x8_R50.pkl')
    #cfg.TRAIN.CHECKPOINT_FILE_PATH = os.path.join(root,'saved_model/kinetics.pkl')
    #slowfast_model.load_state_dict(torch.load(cfg.TRAIN.CHECKPOINT_FILE_PATH), strict=False, encoding='latin1')
    
    '''
    pretrained model
    '''
    
    cu.load_checkpoint(
    cfg.TRAIN.CHECKPOINT_FILE_PATH,
      model,
        False,
          None,
            inflation=False,
              convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
              )    

    return model

class OnlyVideo(nn.Module):
    def __init__(self, device):
        super(OnlyVideo, self).__init__()
        pretrained_model = get_facebook_model(device)
        # ResNet 50
        self.s1 = pretrained_model.s1
        self.s2 = pretrained_model.s2
        self.s3 = pretrained_model.s3
        self.s4 = pretrained_model.s4
        self.s5 = pretrained_model.s5
        self.head = pretrained_model.head
        self.lstm = nn.LSTM(2048, hidden_size=64, num_layers=2, batch_first=False, bidirectional=True, dropout=0.4)
        self.dp = nn.Dropout(0.5)
        self.content_fc = nn.ModuleList([nn.Linear(2048, 4, bias=True) for _ in range(4)])
        self.genre_fc = nn.Linear(2048, 9)
        self.age_fc = nn.Linear(2048, 4)

    def init_hidden(self, batch_size, device):
        hidden = (
                torch.zeros(4, batch_size, 64).requires_grad_().to(device),
                torch.zeros(4, batch_size, 64).requires_grad_().to(device)
        )
        return hidden

    def forward(self, video):
        sequence = []
        for i in range(0, 1024, 64):
            frames = video[:, :, i:i+64, :, :]
            x = [frames]
            x = self.s1(x)
            x = self.s2(x)
            x = self.s3(x)
            x = self.s4(x)
            x = self.s5(x)
            x = self.head(x)
            sequence.append(x)
        sequence = torch.stack(sequence)
        hidden = self.init_hidden(video.size(0),video.device)
        features, _ = self.lstm(sequence, hidden)
        features = features.view(video.size(0), -1)
        features = self.dp(features)
        content = []
        for fc in self.content_fc:
            content.append(fc(features))
        content = torch.stack(content)
        genre = self.genre_fc(features)
        age = self.age_fc(features)
        return content, genre, age


        

if __name__=='__main__':
    #get_slowfast(torch.device('cuda:1'))
    device = torch.device('cuda:1')
    model = OnlyVideo(device).to(device)
    x = torch.rand(1,3,1024,112,112).to(device)
    print(model(x)[0].shape)
