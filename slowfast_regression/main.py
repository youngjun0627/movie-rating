import os
import time
import numpy as np
import torch
import argparse
from torch import nn, optim
from torch.utils.data import DataLoader
from MODELS import slowfastnet
from MODELS.x3d import generate_model
from DATASET.dataset import VideoDataset
from UTILS.loss import Custom_CrossEntropyLoss, Custom_MSELoss
from DATASET.TRANSFORMS.transform import create_train_transform, create_val_transform
from tensorboardX import SummaryWriter
from config import params
from torch.backends import cudnn
from activate import train, val

def save_model(model, optimizer, scheduler, epoch):
    model_cpu = model.to('cpu')
    state = {
            'model' : model_cpu.state_dict(),
            'optimizer':optimizer.state_dict(),
            'scheduler' : scheduler.state_dict()
            }
    if not (os.path.isdir('saved_model')) : os.mkdir('./saved_model')
    torch.save(state,'./saved_model/saved_model_{}.pth'.format(epoch+1))


def main():
    cudnn.benchmark = False
    device = torch.device('cuda:{}'.format(params['gpu'][0]))
    print(torch.__version__)
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    logdir = os.path.join(params['log'], cur_time)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    writer = SummaryWriter(log_dir=logdir)
    #convert_csv(params['dataset'])
    print('Loading dataset')
    train_transform = create_train_transform(True,True,False,True, size=112, bright=False)
    train_dataset = VideoDataset(params['dataset'],size=112, mode='train', play_time=params['clip_len'],frame_sample_rate=params['frame_sample_rate'], transform=train_transform, sub_classnum = params['num_classes'])
    train_dataloader = DataLoader(
                        train_dataset,
                        batch_size=params['batch_size'],
                        shuffle=True,
                        num_workers=params['num_workers'])

    val_transform = create_val_transform(True,size=112)
    val_dataset = VideoDataset(params['dataset'], size=112,mode='validation', play_time=params['clip_len'],frame_sample_rate=params['frame_sample_rate'], transform = val_transform, sub_classnum = params['num_classes'])
    val_dataloader = DataLoader(
                        val_dataset,
                        batch_size=params['batch_size'],
                        shuffle=False,
                        num_workers=params['num_workers'])

    print('train_dataset : batch_size -> {}, step_size -> {}, frames -> {}'.format(params['batch_size'],len(train_dataloader), params['clip_len']))
    print('validation_dataset : batch_size -> {}, step_size -> {}, frames -> {}'.format(params['batch_size'],len(val_dataloader), params['clip_len']))
    print('=========================================================================================================')
    print('Load model : mode -> {}, label_size -> {}, sub_class_num -> {}'.format(params['mode'], params['label_num'], params['num_classes']))
    
    #model = slowfastnet.resnet50(class_num=params['num_classes'], label_num = params['label_num'], mode = params['mode'])
    ### regression ###
    
    #model = generate_model('XL', n_classes = params['label_num'])
    
    #### class ###
    
    model = generate_model('S', n_classes = params['num_classes'])

    if params['pretrained'] != '':
        pretrained_dict = torch.load(params['pretrained'], map_location='cpu')
        try:
            model_dict = model.module.state_dict()
        except AttributeError:
            model_dict = model.state_dict()
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
        print('load pretrained')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    '''
        ########
        state = torch.load('./saved_model/{}'.format(params['pretrained']))
        model.load_state_dict(state['model'])
    '''


    model = model.to(device)

    ### regression ###
    criterion = Custom_MSELoss(num_classes = params['num_classes']).cuda()
    ### classification ###
    #criterion = nn.CrossEntropyLoss(weight = train_dataset.get_class_weight().to(device))
    #criterion =  Custom_CrossEntropyLoss(weight = train_dataset.get_class_weight().to(device))


    #optimizer = optim.SGD(model.parameters(), lr = params['learning_rate'], momentum = params['momentum'], weight_decay = params['weight_decay'])
    #scheduler = optim.lr_scheduler.StepLR(optimizer,  step_size = params['step'], gamma=0.1)
    optimizer = optim.SGD(model.parameters(),lr = params['learning_rate'],weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience = 7, factor = 0.1, verbose=True)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 30, eta_min = 0)
    model_save_dir = os.path.join(params['save_path'], 'second')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    print("train gogosing")
    for epoch in range(params['epoch_num']):
        train(model, train_dataloader, epoch, criterion, optimizer, writer, device)
        if (epoch+1) % 5==0:
            print('======================================================')
            print('validation gogosing')
            validation_loss= val(model, val_dataloader, epoch, criterion, optimizer, writer, device)
            print('validation loss -> {loss:.5f}'.format(loss = validation_loss))
            #checkpoint = os.path.join(model_save_dir, str(epoch) + '.pth.tar')
            #torch.save(model.state_dict(),checkpoint)
            #save_model(model,optimizer,scheduler,epoch)
            #model = model.to(device)
            print('======================================================')
            scheduler.step(validation_loss)
        #scheduler.step()

    writer.close()


if __name__=='__main__':
    main()
