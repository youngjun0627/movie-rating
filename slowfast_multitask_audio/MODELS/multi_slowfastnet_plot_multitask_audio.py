import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride = 1, downsample=None, head_conv=1):
        super(Bottleneck, self).__init__()
        if head_conv == 1:
            self.conv1 = nn.Conv3d(inplanes,planes, kernel_size = 1, bias = False)
            self.bn1 = nn.BatchNorm3d(planes)
        elif head_conv==3:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size = (3,1,1), bias=False, padding=(1,0,0))
            self.bn1 = nn.BatchNorm3d(planes)
        elif head_conv==5:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size = (5,1,1), bias=False, padding=(2,0,0))
            self.bn1 = nn.BatchNorm3d(planes)
        else:
            raise ValueErrror('Unsupported head_conv!')

        self.conv2 = nn.Conv3d(
                planes, planes, kernel_size = (1,3,3), stride=(1,stride,stride),
                padding=(0,1,1) , bias = False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes*4)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class SlowFast(nn.Module):
    def __init__(self, block = Bottleneck, layers=[3,5,11,7], class_num=4, label_num = 5, dropout=0.5, mode='multi', vocab_size = 0):
        super(SlowFast, self).__init__()
        self.mode = mode
        self.embed_dim = 64
        self.audio_size = 1024
        self.class_num = class_num
        self.label_num = label_num
        self.fast_inplanes = 8
        self.fast_conv1 = nn.Conv3d(3, 8, kernel_size = (5,7,7), stride=(1,2,2), padding=(2,3,3), bias = False)
        self.fast_bn1 = nn.BatchNorm3d(8)
        self.fast_relu = nn.ReLU(inplace=True)
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.fast_res2 = self._make_layer_fast(block, 8, layers[0], head_conv=5)
        self.fast_res3 = self._make_layer_fast(block, 16,layers[1], stride=2, head_conv=5)
        self.fast_res4 = self._make_layer_fast(block, 32, layers[2], stride=2, head_conv=3)
        self.fast_res5 = self._make_layer_fast(block, 64, layers[3], stride=2, head_conv=3)

        self.lateral_p1 = nn.Conv3d(8,8*2,kernel_size = (5,1,1), stride=(8,1,1), bias = False, padding = (2,0,0))
        self.lateral_res2 = nn.Conv3d(32,32*2,kernel_size = (5,1,1), stride=(8,1,1), bias = False, padding = (2,0,0))
        self.lateral_res3 = nn.Conv3d(64,64*2,kernel_size = (5,1,1), stride=(8,1,1), bias = False, padding = (2,0,0))
        self.lateral_res4 = nn.Conv3d(128,128*2,kernel_size = (5,1,1), stride=(8,1,1), bias = False, padding = (2,0,0))


        self.slow_inplanes = 64+64//8*2
        self.slow_conv1 = nn.Conv3d(3,64,kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3), bias=False)
        self.slow_bn1 = nn.BatchNorm3d(64)
        self.slow_relu = nn.ReLU(inplace=True)
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.slow_res2 = self._make_layer_slow(block, 64, layers[0], head_conv=3)
        self.slow_res3 = self._make_layer_slow(block, 128, layers[1], stride=2,head_conv=3)
        self.slow_res4 = self._make_layer_slow(block, 256, layers[2],stride=2,head_conv=3)
        self.slow_res5 = self._make_layer_slow(block,512,layers[3], stride=2, head_conv=3)
        
        self.dp = nn.Dropout(dropout)

        self.classifier1 = nn.ModuleList([nn.Linear(self.fast_inplanes+2048+self.embed_dim+self.audio_size, 256) for _ in range(label_num)])
        self.classifier2 = nn.ModuleList([nn.Linear(256, class_num) for _ in range(label_num)])
        
        #self.classifier = nn.Linear(self.fast_inplanes + 2048 + self.embed_dim + self.audio_size, class_num * label_num)
        # text #
        self.embedding = nn.EmbeddingBag(vocab_size, self.embed_dim)
        self.textfc1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.text_init_weights()

        # genre fc, age fc #
        self.genrefc = nn.Linear(self.fast_inplanes + 2048 + self.embed_dim + self.audio_size, 9)
        self.agefc = nn.Linear(self.fast_inplanes + 2048 + self.embed_dim + self.audio_size, 4)

        self.extract_audio = nn.Sequential(nn.Conv2d(1, 64, kernel_size=(1,5), stride=(1,2)),\
                                        nn.LeakyReLU(),
                                        nn.MaxPool2d((1,2)),
                                        nn.Conv2d(64, 128, kernel_size=(1,5), stride=(1,2), padding=(1,1)),
                                        nn.LeakyReLU(),
                                        nn.MaxPool2d((1,2)),
                                        nn.Conv2d(128, 256, kernel_size=(1,5), stride=(1,2), padding=(1,1)),
                                        nn.LeakyReLU(),
                                        nn.MaxPool2d((1,2)),
                                        nn.Conv2d(256, 512, kernel_size=3, stride=(1,2), padding=(1,1)),
                                        nn.LeakyReLU(),
                                        nn.MaxPool2d(2),
                                        nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=(1,1)),
                                        nn.AvgPool2d((22,16)))
    def text_init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.textfc1.weight.data.uniform_(-initrange, initrange)
        self.textfc1.bias.data.zero_()

    

    def forward(self, input, text, offset, audio):
        fast, lateral = self.FastPath(input[:,:,::1,:,:])
        slow = self.SlowPath(input[:,:,::8,:,:],lateral)
        video = torch.cat([slow,fast],dim=1)
        text = self.embedding(text,offset)
        text = self.textfc1(text)
        audio = self.extract_audio(audio)
        audio = audio.view(audio.shape[0], -1)
        features = torch.cat([video, text, audio], dim=1)
        features = self.dp(features)
        # method 1
        outputs = []
        for fc1, fc2 in zip(self.classifier1, self.classifier2):
            outputs.append(fc2(self.dp(fc1(features))))
        outputs = torch.stack(outputs)
        '''
        # method 2
        outputs = self.classifier(features)
        outputs = outputs.view(self.label_num, -1, self.class_num)
        '''
        genre = self.genrefc(features)
        age = self.agefc(features)
        return outputs, genre, age
        '''
        outputs = []
        if self.mode=='multi':
            fcs = self.multi_fcs
        elif self.mode == 'single':
            fcs = self.single_fcs
        for fc in fcs:
            for i, dp in enumerate(self.dps):
                if i==0:
                    output = fc(dp(x))
                else:
                    output += fc(dp(x))
            else:
                output/=len(self.dps)
            outputs.append(output)
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1,0,2)
        return outputs
        '''
    def SlowPath(self,input,lateral):
        x = self.slow_conv1(input)
        #print('slow1',x.shape)
        x= self.slow_bn1(x)
        #print('slow2',x.shape)
        x = self.slow_relu(x)
        #print('slow3',x.shape)
        x = self.slow_maxpool(x)
        #print('slow4',x.shape)
        x = torch.cat([x,lateral[0]],dim=1)
        #print('slow5',x.shape)
        x = self.slow_res2(x)
        #print('slow6',x.shape)
        x = torch.cat([x,lateral[1]],dim=1)
        #print('slow7',x.shape)
        x = self.slow_res3(x)
        #print('slow8',x.shape)
        x = torch.cat([x,lateral[2]],dim=1)
        #print('slow9',x.shape)
        x = self.slow_res4(x)
        #print('slow10',x.shape)
        x = torch.cat([x,lateral[3]],dim=1)
        #print('slow11',x.shape)
        x = self.slow_res5(x)
        #print('slow12',x.shape)
        x = nn.AdaptiveAvgPool3d(1)(x)
        #print('slow13',x.shape)
        x = x.view(-1,x.size(1))
        #print('slow14',x.shape)
        return x

    def FastPath(self,input):
        lateral = []
        x = self.fast_conv1(input)
        #print('fast1', x.shape)
        x = self.fast_bn1(x)
        x = self.fast_relu(x)
        pool1 = self.fast_maxpool(x)
        lateral_p = self.lateral_p1(pool1)
        lateral.append(lateral_p)
        #print('fast2', pool1.shape)
        res2 = self.fast_res2(pool1)
        #print('res2', res2.shape)
        lateral_res2 = self.lateral_res2(res2)
        #print('lateral_res2', lateral_res2.shape)
        lateral.append(lateral_res2)
    
        res3 = self.fast_res3(res2)
        #print('res3', res3.shape)
        lateral_res3 = self.lateral_res3(res3)
        #print('lateral_res3', lateral_res3.shape)
        lateral.append(lateral_res3)

        res4 = self.fast_res4(res3)
        #print('res4', res4.shape)
        lateral_res4 = self.lateral_res4(res4)
        #print('lateral_res4', lateral_res4.shape)
        lateral.append(lateral_res4)

        res5 = self.fast_res5(res4)
        #print('res5', x.shape)
        x = nn.AdaptiveAvgPool3d(1)(res5)
        #print('x', x.shape)
        x = x.view(-1,x.size(1))
        #print('x', x.shape)

        return x, lateral

    def _make_layer_fast(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.fast_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.fast_inplanes,
                    planes * block.expansion,
                    kernel_size = 1,
                    stride = (1,stride,stride),
                    bias = False),
                nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(self.fast_inplanes,planes,stride,downsample, head_conv=head_conv))
        self.fast_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.fast_inplanes,planes, head_conv=head_conv))
        return nn.Sequential(*layers)


    def _make_layer_slow(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.slow_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.slow_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1,stride,stride),
                    bias = False), 
                nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(self.slow_inplanes, planes, stride, downsample, head_conv = head_conv))
        self.slow_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.slow_inplanes, planes, head_conv=head_conv))

        self.slow_inplanes = planes * block.expansion + planes*block.expansion//8*2
        return nn.Sequential(*layers)

def resnet50(**kwargs):
    model = SlowFast(Bottleneck,[3,4,6,3],**kwargs)
    return model

def resnet101(**kwargs):
    model = SlowFast(Bottleneck, [3,4,23,3], **kwargs)
    return model

def c2_msra_fill(module: nn.Module) -> None:
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity='relu')
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)

def init_weights(model, fc_init_std=0.01, zero_init_final_bn=True):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            c2_msra_fill(m)
        elif isinstance(m, nn.BatchNorm3d):
            batchnorm_weight=1.0
            if m.weight is not None:
                m.weight.data.fill_(batchnorm_weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std = fc_init_std)
            if m.bias is not None:
                m.bias.data.zero_()

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

