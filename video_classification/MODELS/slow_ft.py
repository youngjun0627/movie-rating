'''
Here's the snippet from my code.
Also modify the .yaml config file as needed.
Hope this helps! 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pytorchvideo

class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(SelfAttention,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out, attention

class OnlyVideo(nn.Module):
    def __init__(self, device):
        super(OnlyVideo, self).__init__()
        #pretrained_model = get_facebook_model(device)
        # ResNet 50
        model_name = "slow_r50"
        model = torch.hub.load("facebookresearch/pytorchvideo:main", model_name, pretrained=True)
        self.extractor = nn.Sequential(*list(*model.children())[:-1]) # num_feature=2304
        self.avgpool = nn.AvgPool3d(kernel_size=(8, 7, 7), stride=(1, 1, 1), padding=(0, 0, 0))
        self.attention = SelfAttention(2048)

        self.init_freeze()
        self.lstm = nn.LSTM(1024, hidden_size=64, num_layers=2, batch_first=False, bidirectional=True, dropout=0.4)
        self.dp = nn.Dropout(0.5)
        self.content_fc = nn.ModuleList([nn.Linear(2048, 4, bias=True) for _ in range(4)])
        self.genre_fc = nn.Linear(2048, 9)
        self.age_fc = nn.Linear(2048, 4)
        self.adapt_avgpool = nn.AdaptiveAvgPool3d(output_size=1)
        self.encoder = nn.Linear(2048, 1024, bias=True)

    def init_freeze(self):
        for param in self.extractor.parameters():
            param.requires_grad = False
    
    def init_hidden(self, batch_size, device):
        hidden = (
                torch.zeros(4, batch_size, 64).requires_grad_().to(device),
                torch.zeros(4, batch_size, 64).requires_grad_().to(device)
        )
        return hidden

    def forward(self, video):
        sequence = []
        for i in range(0, 1024, 64):
            x = video[:, :, i:i+64:8, :, :]
            x = self.extractor(x)
            x = self.avgpool(x)
            x, _ = self.attention(x.squeeze(2))  # remove time dimension
            x = x.view(x.size(0), -1)
            x = self.encoder(x)
            sequence.append(x)
        sequence = torch.stack(sequence)
        hidden = self.init_hidden(video.size(0),video.device)
        features, _ = self.lstm(sequence, hidden)
        print(features.shape)
        features = features.view(video.size(0), -1)
        print(features.shape)
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
