import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Only_Audio(nn.Module):
    def __init__(self,  class_num=4, label_num = 5, dropout=0.5, mode='multi', vocab_size = 0):
        super(Only_Audio, self).__init__()
        self.mode = mode
        self.audio_size = 1024
        self.class_num = class_num
        self.label_num = label_num
        
        self.dp = nn.Dropout(dropout)
        self.classifier1 = nn.ModuleList([nn.Linear(self.audio_size, class_num) for _ in range(label_num)])
        # genre fc, age fc #
        self.genrefc = nn.Linear(self.audio_size, 9)
        self.agefc = nn.Linear(self.audio_size, 4)

        self.extract_audio = nn.Sequential(nn.Conv2d(1, 32, kernel_size=(3,15), stride=(1,3), padding=(1,1)),\
                                        nn.BatchNorm2d(32),\
                                        nn.LeakyReLU(inplace = True),\
                                        nn.MaxPool2d(2),\
                                        nn.Dropout(0.3),
                                        nn.Conv2d(32, 64, kernel_size=(3,15), stride=(1,3), padding=(1,1)),\
                                        nn.BatchNorm2d(64),\
                                        nn.LeakyReLU(inplace = True),\
                                        nn.MaxPool2d(2),\
                                        nn.Dropout(0.3),\
                                        nn.Conv2d(64, 128, kernel_size=(3,15), stride=(1,3), padding=(1,1)),\
                                        nn.BatchNorm2d(128),\
                                        nn.LeakyReLU(inplace = True),\
                                        nn.MaxPool2d(2),\
                                        nn.Dropout(0.3),
                                        nn.Conv2d(128, 256, kernel_size=(3,11), stride=(1,3), padding=(1,1)),\
                                        nn.AdaptiveAvgPool2d(2)\
                                        )
        self.audio_init_weights()
        
    
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

    def forward(self, audio):
        audio = self.extract_audio(audio)
        features = audio.view(audio.shape[0], -1)
        
        outputs = []
        for fc in self.classifier1:
            outputs.append(fc(self.dp(features)))
        outputs = torch.stack(outputs)
        genre = self.genrefc(self.dp(features))
        age = self.agefc(self.dp(features))
        return outputs, genre, age

if __name__=='__main__':
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    num_classes = 6
    num_label = 7
    mode = 'multi'
    #model = resnet50(class_num = num_classes, label_num = num_label, mode = mode).cuda()
    #model = generate_model('M').cuda()
    #for _ in range(2):
    #    input_tensor = torch.autograd.Variable(torch.rand(1,3,100,224,224)).cuda()
    #    output = model(input_tensor)
    #    print(output.size())
    #num_classes = 6
    #num_label = 7
    #model = resnet50(class_num = 1, label_num = num_label, mode='single').cuda()
    #for _ in range(2):
    #    input_tensor = torch.autograd.Variable(torch.rand(1,3,1000,112,112)).cuda()
    #    output = model(input_tensor)
    #    print(output.size())

