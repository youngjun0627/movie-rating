import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def mish(x):
    return x * torch.tanh(F.softplus(x))

class Only_Audio(nn.Module):
    def __init__(self,  class_num=4, label_num = 5, dropout=0.5, mode='multi', vocab_size = 0):
        super(Only_Audio, self).__init__()
        self.mode = mode
        self.audio_size = 1000
        self.class_num = class_num
        self.label_num = label_num
        feat_s = 12000 // self.audio_size
        self.feature_size = int((feat_s*2-1) * 256 / 2)
        self.dp = nn.Dropout(dropout)
        self.fc = nn.Linear(self.feature_size, 1024)
        self.contentfc1 = nn.ModuleList([nn.Linear(1024, 256) for _ in range(label_num)])
        self.contentfc2 = nn.ModuleList([nn.Linear(256, class_num) for _ in range(label_num)])
        # genre fc, age fc #
        self.genrefc = nn.Linear(1024, 9)
        self.agefc = nn.Linear(1024, 4)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,7), stride=(1,3), padding=(1,1))
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,7), stride=(1,3), padding=(1,1))
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,5), stride=(1,1), padding=(1,1))
        self.bn3 = nn.BatchNorm2d(128)
        self.maxpool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3,5), stride=(1,1), padding=(1,1))
        self.bn4 = nn.BatchNorm2d(256)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        '''
        self.extract_audio = nn.Sequential(nn.Conv2d(1, 32, kernel_size=(3,7), stride=(1,3), padding=(1,1)),\
                                        nn.BatchNorm2d(32),\
                                        nn.LeakyReLU(inplace = True),\
                                        nn.MaxPool2d(2),\
                                        nn.Conv2d(32, 64, kernel_size=(3,7), stride=(1,3), padding=(1,1)),\
                                        nn.BatchNorm2d(64),\
                                        nn.LeakyReLU(inplace = True),\
                                        nn.MaxPool2d(2),\
                                        nn.Conv2d(64, 128, kernel_size=(3,5), stride=(1,1), padding=(1,1)),\
                                        nn.BatchNorm2d(128),\
                                        nn.LeakyReLU(inplace = True),\
                                        nn.MaxPool2d(2),\
                                        nn.Conv2d(128, 256, kernel_size=(3,5), stride=(1,1), padding=(1,1)),\
                                        nn.AdaptiveAvgPool2d(1)\
                                        #nn.AdaptiveAvgPool2d(2)\
                                        )
        '''
        #self.audio_init_weights()

        self.lstm = nn.LSTM(256, 64, num_layers = 2, dropout = 0.3, batch_first = False, bidirectional=True)
        

    '''
    def audio_init_weights(self):
        for m in self.extract_audio:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)


            elif isinstance(m, nn.BatchNorm2d):
                batchnorm_weight=1.0
                if m.weight is not None:
                    m.weight.data.fill_(batchnorm_weight)
                if m.bias is not None:
                    m.bias.data.zero_()
    '''

    def init_hidden(self, batch_size, device):
      return (
          torch.zeros(4, batch_size, 64).requires_grad_().to(device),\
          torch.zeros(4, batch_size, 64).requires_grad_().to(device)\
      )

    def mish(self, x):
        return x * torch.tanh(F.softplus(x))

    def extract_audio(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool1(x)
        x = self.mish(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.maxpool2(x)
        x = self.mish(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.maxpool3(x)
        x = self.mish(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.avgpool(x)
        return x

    def forward(self, audio):
        features = []
        start = 0
        while start + self.audio_size <= audio.size(3):
            feature = self.extract_audio(audio[:, :, :, start:start+self.audio_size])
            features.append(feature)
            start += self.audio_size//2
        '''
        for start in range(self.audio_size//2)
        for i in range(audio.size(3)//self.audio_size):
            feature = self.extract_audio(audio[:, :, :, i:i+self.audio_size])
            features.append(feature)
        '''
        features = torch.stack(features)
        features = features.view(features.size(0), features.size(1), -1)
        hidden = self.init_hidden(features.size(1), features.device)
        features, _ = self.lstm(features, hidden)
        features = features.view(features.size(1), -1)
        features = self.dp(self.mish(self.fc(features)))
        outputs = []
        for fc1, fc2 in zip(self.contentfc1, self.contentfc2):
            outputs.append(fc2(self.dp(self.mish(fc1(features)))))
        outputs = torch.stack(outputs)
        genre = self.genrefc(features)
        age = self.agefc(features)

        return outputs, genre, age

if __name__=="__main__":
    model = Only_Audio().cuda()
    data = torch.randn(4,1,40,12000).cuda()
    model(data)
