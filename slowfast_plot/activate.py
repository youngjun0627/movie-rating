import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from MODELS import slowfastnet
from tensorboardX import SummaryWriter
from UTILS.metrics import custom_metric
from sklearn.metrics import f1_score
import torch.nn.functional as F

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum=0
        self.count=0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count+=n
        self.avg = self.sum/self.count

def train(model, train_dataloader, epoch, criterion, optimizer, writer, device, mode = 'single', label_num=1, display=1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    model.train()
    running_loss = 0
    label_dic = {0:'sex_nudity', 1:'violence_gore', 2:'alcohol_drugs_smoking', 3:'frightening_intense_scene'}
    preds=None
    targets = None
    if mode=='single':
        preds=[]
        targets = []
    elif mode=='multi':
        preds =[[] for _ in range(label_num)]
        targets = [[] for _ in range(label_num)]
    else:
        raise Exception('{} is not supported mode'.format(mode))
    for step, (inputs, labels) in enumerate(train_dataloader):
        data_time.update(time.time()-end)
        inputs=inputs.to(device)
        #print(inputs.shape)
        labels = labels.long().to(device)
        outputs = model(inputs)
        #print(outputs.shape)
        #print(labels.shape)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #losses.update(loss.item(),inputs.size(0))
        running_loss+=loss.item()
        with torch.no_grad():
            if mode=='single':
                for i in range(outputs.shape[0]):
                    preds.append(np.argmax(F.softmax(outputs[i], dim=0).cpu().detach().numpy()))
                    targets.append(labels[i].cpu().detach().numpy())
            elif mode=='multi':
                labels=torch.transpose(labels, 0, 1) # B, L -> L, B
                for idx, (output, label) in enumerate(zip(outputs, labels)):
                    for i in range(output.shape[0]):
                        preds[idx].append(np.argmax(F.softmax(output[i], dim=0).cpu().detach().numpy()))
                        targets[idx].append(label[i].cpu().detach().numpy())
            else:
                raise Exception('{} is not supported mode'.format(mode))
        batch_time.update(time.time()-end)
        end = time.time()
        if (step+1)% display==0:
            print('----------------------------------------------')
            for param in optimizer.param_groups:
                print('lr : {}'.format(param['lr']))
            print_string = 'Epoch:[{0}] step : [{1}/{2}]'.format(epoch+1, step+1, len(train_dataloader))
            print(print_string)
            print_string = 'data_time: {data_time:.3f}, batch_time: {batch_time:.3f}'.format(data_time = data_time.val, batch_time = batch_time.val)
            print(print_string)
            if mode=='single':
                print_string = 'Loss : {loss:.5f} \t F1_score : {metric:.5f}'.format(loss=running_loss/display, metric = f1_score(np.array(targets), np.array(preds), average='macro'))
                print(print_string)
                targets.clear()
                preds.clear()
            elif mode=='multi':
                print_string = 'Loss : {loss:.5f}'.format(loss=running_loss/display)
                print(print_string)
                for idx, (pred, target) in enumerate(zip(preds, targets)):
                    score = f1_score(np.array(target), np.array(pred), average='macro')
                    print_string = 'Label {label_name} -> F1_score : {metric:.5f}'.format(label_name=label_dic[idx], metric=score)
                    print(print_string)
                    target.clear()
                    pred.clear()

            running_loss = 0
            
            #print(outputs)
            #print(labels)

        #writer.add_scalar('train_loss_epoch', losses.avg, epoch)

def val(model, val_dataloader, epoch, criterion, optimizer, writer, device, mode = 'single', label_num=1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.eval()
    end = time.time()
    label_dic = {0:'sex_nudity', 1:'violence_gore', 2:'alcohol_drugs_smoking', 3:'frightening_intense_scene'}
    preds=None
    targets = None
    if mode=='single':
        preds=[]
        targets = []
    elif mode=='multi':
        preds =[[] for _ in range(label_num)]
        targets = [[] for _ in range(label_num)]
    else:
        raise Exception('{} is not supported mode'.format(mode))
    running_loss = 0
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(val_dataloader):
            data_time.update(time.time()-end)
            inputs=inputs.to(device)
            labels = labels.long().to(device)
            outputs = model(inputs)
            #print(outputs)
            #print(labels)
            loss = criterion(outputs, labels)
            #print(loss.item())
            running_loss+=loss.item()
            #print(outputs.shape, labels.shape)
            
            if mode=='single':
                for i in range(outputs.shape[0]):
                    preds.append(np.argmax(F.softmax(outputs[i], dim=0).cpu().detach().numpy()))
                    targets.append(labels[i].cpu().detach().numpy())
            elif mode=='multi':
                labels=torch.transpose(labels, 0, 1) # B, L -> L, B
                for idx, (output, label) in enumerate(zip(outputs, labels)):
                    for i in range(output.shape[0]):
                        preds[idx].append(np.argmax(F.softmax(output[i], dim=0).cpu().detach().numpy()))
                        targets[idx].append(label[i].cpu().detach().numpy())
            batch_time.update(time.time()-end)

            end = time.time()

        result = 0
        if mode=='single':
            score = f1_score(np.array(targets), np.array(preds), average='macro')
            result = score
            print_string = 'Loss : {loss:.5f} \t F1_score : {metric:.5f}'.format(loss=running_loss/len(val_dataloader), metric = score)
            print(print_string)
            result=score
        elif mode=='multi':
            print_string = 'Loss : {loss:.5f}'.format(loss=running_loss/len(val_dataloader))
            print(print_string)
            result = 0 
            for idx, (pred, target) in enumerate(zip(preds, targets)):
                score = f1_score(np.array(target), np.array(pred), average='macro')
                result += score
                print_string = 'Label {label_name} -> F1_score : {metric:.5f}'.format(label_name=label_dic[idx], metric=score)
                print(print_string)
            result/=label_num
        return result
        #return running_loss/len(val_dataloader), f1_score(np.array(targets), np.array(preds), average='macro')

