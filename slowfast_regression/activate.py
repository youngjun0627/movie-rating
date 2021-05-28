import os
import time
import numpy as np
from config import params
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

def train(model, train_dataloader, epoch, criterion, optimizer, writer, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    model.train()
    running_loss = 0
    for step, (inputs, labels) in enumerate(train_dataloader):
        data_time.update(time.time()-end)
        inputs=inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs) 
        outputs = outputs.squeeze(1)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #losses.update(loss.item(),inputs.size(0))
        running_loss+=loss.item()
        batch_time.update(time.time()-end)
        end = time.time()
        if (step+1)% params['display']==0:
            print('----------------------------------------------')
            for param in optimizer.param_groups:
                print('lr : {}'.format(param['lr']))
            print_string = 'Epoch:[{0}] step : [{1}/{2}]'.format(epoch+1, step+1, len(train_dataloader))
            print(print_string)
            print_string = 'data_time: {data_time:.3f}, batch_time: {batch_time:.3f}'.format(data_time = data_time.val, batch_time = batch_time.val)
            print(print_string)
            print_string = 'Loss : {loss:.5f} '.format(loss=running_loss/params['display'])
            running_loss = 0
            print(print_string)
        #writer.add_scalar('train_loss_epoch', losses.avg, epoch)

def val(model, val_dataloader, epoch, criterion, optimizer, writer, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.eval()
    end = time.time()
    preds = []
    targets = []
    running_loss = 0
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(val_dataloader):
            data_time.update(time.time()-end)
            inputs=inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze(1)

            loss = criterion(outputs, labels)
            running_loss+=loss.item()

            preds.append(outputs)
            targets.append(labels)
            batch_time.update(time.time()-end)
            end = time.time()
        for pred, target in zip(preds,targets):
            print(pred, target)
        return running_loss/len(val_dataloader)

