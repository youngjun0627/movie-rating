import torch.nn as nn
import torch

class ResNet(nn.Module):

    def __init__(self, n_classes=4, label_num = 5):

        super(ResNet, self).__init__()

        self.extract_audio = nn.Sequential(nn.Conv2d(1, 64, kernel_size=(1,4), stride=(1,2)),\
                                        nn.LeakyReLU(),\
                                        nn.MaxPool2d((1,2)),\
                                        nn.Conv2d(64, 128, kernel_size=(1,4), stride=(1,2), padding=(0,1)),\
                                        nn.LeakyReLU(),\
                                        nn.MaxPool2d((1,2)),\
                                        nn.Conv2d(128, 256, kernel_size=(1,4), stride=(1,2), padding=(0,1)),\
                                        nn.LeakyReLU(),\
                                        nn.MaxPool2d((1,2)),\
                                        nn.Conv2d(256, 512, kernel_size=3, stride=(1,2), padding=(0,1)),\
                                        nn.LeakyReLU(),\
                                        nn.MaxPool2d(2),\
                                        nn.Conv2d(512,1024,kernel_size =3, stride=1, padding=(0,1)),\
                                        nn.AvgPool2d(2))
        self.dropout = nn.Dropout(0.5)
        self.fcs = nn.ModuleList([nn.Linear(1024*8*8, n_classes) for _ in range(label_num)])
        self.genrefc = nn.Linear(1024*8*8, 9)
        self.agefc = nn.Linear(1024*8*8, 4)

    def forward(self, x):
        x = self.extract_audio(x)
        x = x.view(-1, 1024*8*8)
        x = self.dropout(x)
        outputs = []
        for fc in self.fcs:
            outputs.append(fc(x))
        outputs = torch.stack(outputs)

        genre = self.genrefc(x)
        age = self.agefc(x)
        return outputs, genre, age
