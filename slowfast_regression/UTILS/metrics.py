import numpy as np
import torch
from sklearn.metrics import f1_score as f1


def get_metrix(preds, targets):
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    #assert preds.shape[0]==targets.shape[0] and preds.shape[1]==targets.shape[1]

    label_size = 3
    metrix = np.zeros((label_size,label_size))

    for target,pred in zip(targets, preds): # batch
        metrix[target][np.argmax(pred)]+=1
            
    return metrix

def precision_recall(metrix):
    class_num = len(metrix)
    dic = {}
    precision = 0
    recall = 0

    for i in range(class_num):
        dic[i]={'TP':0,'TN':0,'FN':0} # order of list is TP, TN, FN 
    for i in range(class_num):
        for j in range(class_num):
            if i==j:
                dic[i]['TP'] = metrix[i][j]
            else:
                dic[i]['FN'] = metrix[i][j]
                dic[i]['TN'] = metrix[j][i]
    for _, value in dic.items():
        TP = value['TP']
        FN = value['FN']
        TN = value['TN']
        if TP==0:
            continue
        precision += TP/(TP+TN)
        recall += TP/(TP+FN)
    return precision/class_num, recall/class_num

def f1_score(precision, recall):
    if precision==0 and recall==0:
        return 0
    return 2*(precision*recall)/(precision+recall)

def metric(preds,targets):
    metrix = get_metrix(preds, targets)
    precision, recall = precision_recall(metrix)
    score = f1_score(precision, recall)
    return score

def custom_metric(preds, targets):
    list_len = len(preds)
    class_num = 4
    print(preds.shape, targets.shape)
    preds = np.transpose(preds,(1,0,2))
    targets = np.transpose(targets, (1,0))
    label_size = preds.shape[0]
    result=0

    for label in range(label_size):
        _preds = preds[label]
        _targets = targets[label]
        matrix = np.zeros((class_num, class_num))

        for p, t in zip(_preds, _targets):
            matrix[t][np.argmax(p)]+=1
        precision, recall = precision_recall(matrix)
        score = f1_score(precision, recall)
        result+=score

    return result/label_size



if __name__=='__main__':

    preds_list = []
    targets_list = []
    for _ in range(100):
        preds = torch.rand(4,7,3).cuda()
        targets = torch.randint(2,size=(4,7)).cuda()
        for i in range(preds.shape[0]):
            preds_list.append(preds[i].cpu().detach().numpy())
            targets_list.append(targets[i].cpu().detach().numpy())
    print(custom_metric(np.array(preds_list), np.array(targets_list)))
    #a = 0
    #for i in range(7):
        #b =  f1(preds[i].detach().cpu().numpy(),targets[i].detach().cpu().numpy())
        #a+=b
