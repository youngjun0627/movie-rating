from sklearn.utils import class_weight
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss

class LabelSmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.25):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        
    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                                    .fill_(smoothing / (n_classes - 1)) \
                                    .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

                                                                                                                    
    def forward(self, inputs, targets):
        targets = LabelSmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                self.smoothing)
        lsm = F.log_softmax(inputs, -1)
        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)
        loss = -(targets * lsm).sum(-1)
        if self.reduction == 'sum':
            loss = loss.sum()
        
        elif self.reduction == 'mean':
            loss = loss.mean()
        
        return loss

class Custom_CrossEntropyLoss(LabelSmoothCrossEntropyLoss):
    def __init__(self, num_classes=4, weight = None):
        super(Custom_CrossEntropyLoss,self).__init__()
        self.num_classes = num_classes
        self.weight = weight

    def forward(self,output, label):
        celoss = LabelSmoothCrossEntropyLoss(weight = self.weight, reduction = 'mean')
        #print(output,label)
        #print(output.shape, label.shape)
        #output = output.squeeze()
        #label = label.squeeze()
        #print(output.shape, label.shape)
        loss = celoss(output,label)
        return loss

class Custom_MultiCrossEntropyLoss(LabelSmoothCrossEntropyLoss):
    def __init__(self, num_classes = 4, label_num = 1, weight = None):
        super(Custom_MultiCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.label_num = label_num
        self.celoss = None
        if label_num==1:
            self.celoss = LabelSmoothCrossEntropyLoss(weight = weight, reduction = 'mean')
        else:
            self.celoss = []
            for i in range(label_num):
                self.celoss.append(LabelSmoothCrossEntropyLoss(weight = weight[i], reduction='mean'))

    def forward(self, output, label):
        #celoss = LabelSmoothCrossEntropyLoss(weight = self.weight, reduction = 'mean')
        loss=0
        label = torch.transpose(label,0,1)
        for idx, (pred, target) in enumerate(zip(output, label)):
            #print(pred.shape)
            #print(target.shape)
            loss+=self.celoss[idx](pred,target)
        return loss/self.label_num

class Custom_BCELoss(nn.BCELoss):
    def __init__(self):
        super(Custom_BCELoss,self).__init__()


        #train_dataset.labels # size (video_size, 7)
        

    def forward(self, output, label):
        dic = {0:0,1:0,2:0,3:1,4:1,5:1}
        print(output.shape)
        for b in range(label.shape[0]):
            for c in range(label.shape[1]):
                label[b][c] = dic[label[b][c].item()]
        print(label)
        bceloss = nn.BCEWithLogitsLoss()

        loss = 0
        label = torch.transpose(label,0,1)
        print(label)
        for i in range(7):
            loss+=bceloss(output,label)
        loss /= 7
        return loss

class Custom_MSELoss(nn.MSELoss):
    def __init__(self,num_classes = 3):
        super(Custom_MSELoss,self).__init__()
        self.num_classes = num_classes

    def forward(self, output, label):
        #output = torch.sigmoid(output*0.5)
        #label /= self.num_classes
        output = output.squeeze()
        label = label.squeeze()
        label = label.float()
        #print(output, label)
        #smooth_l1_loss = nn.SmoothL1Loss()
        #loss = smooth_l1_loss(output,label)
        mseloss = nn.MSELoss()
        #loss = mseloss(output,label)
        loss = torch.sqrt(mseloss(output,label))
        return loss

        

class Custom_HingeLoss(nn.MultiMarginLoss):
        def __init__(self, p=1, margin=1, weight=None, size_average=True, label_num = 7):

            super(Custom_HingeLoss, self).__init__()
            self.p=p
            self.margin=margin
            self.weight=weight
            self.size_average=size_average
            self.label_num = label_num

        def forward(self, preds, targets):
            targets = torch.transpose(targets,0,1)
            svm = nn.MultiMarginLoss(p=self.p, margin=self.margin, weight=self.weight, size_average=self.size_average)
            loss=0
            for i in range(self.label_num):
                loss+=svm(preds[i],targets[i])

            return loss/self.label_num

if __name__=='__main__':
    #a = torch.rand(2,7,3)
    #b = torch.randint(3,size=(2,7))
    a =[[[ 0.0023],
        [-0.0153],
        [ 0.0173],
        [ 0.0088],
        [-0.0095]],

        [[ 0.0023],
            [-0.0154],
            [ 0.0172],
            [ 0.0085],
            [-0.0097]],
        
        [[ 0.0022],
            [-0.0150],
            [ 0.0174],
            [ 0.0095],
        [-0.0086]],
        
        [[ 0.0023],
            [-0.0152],
            [ 0.0173],
            [ 0.0091],
            [-0.0092]],

        
        [[ 0.0022],
            [-0.0152],
            [ 0.0173],
            [ 0.0090],
            [-0.0092]],
        
        [[ 0.0023],
            [-0.0149],
            [ 0.0174],
            [ 0.0098],
            [-0.0084]],
        
        [[ 0.0023],
            [-0.0154],
            [ 0.0172],
            [ 0.0086],
            [-0.0096]],
        
        [[ 0.0022],
            [-0.0151],
            [ 0.0173],
            [ 0.0092],
            [-0.0090]]]
    b = [[2, 2, 2, 2, 2],
                    [1, 3, 3, 1, 3],
                            [2, 2, 2, 2, 2],
                                    [2, 0, 2, 2, 2],
                                            [2, 1, 2, 1, 1],
                                                    [2, 3, 1, 2, 3],
                                                            [3, 3, 3, 3, 3],
                                                                    [3, 1, 3, 1, 1]]


    a = [[[1],[1],[1],[1],[1]],
            [[0],[0],[0],[0],[0]]]
    b = [[1,0,0,1,0],[0,1,0,2,1]]
    a = torch.tensor(a)
    b = torch.tensor(b)
    f = Custom_MSELoss()
    #print(Loss(a,b))
    #f = Custom_CrossEntropyLoss()
    for i in range(7):
        print(f(a,b))
