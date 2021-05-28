import os
import time
import numpy as np
import torch
import argparse
from torch import nn, optim
from torch.utils.data import DataLoader
from MODELS import slowfastnet, x3d, multi_slowfastnet, multi_x3d, efficientnet, multi_x3d_plot
from DATASET.dataset import VideoDataset
from UTILS.loss import Custom_CrossEntropyLoss, Custom_MSELoss, Custom_MultiCrossEntropyLoss
from DATASET.TRANSFORMS.transform import create_train_transform, create_val_transform
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from activate import train, val


#from CONFIG.x3d_single import params
from CONFIG.x3d_multi import params
#from CONFIG.efficientnet3D_b0_multi import params
#from CONFIG.efficientnet3D_b2_multi import params
from CONFIG.slowfast_multi import params
#from CONFIG.slowfast_multi_v2 import params

#parser = argparse.ArgumentParser(description='configuration')
#parser.add_argument('--config', required=True, help='Insert config')
#if parser.config=='x3d_multi':
#    from CONFIG.x3d_multi import params

print(os.getpid())
def save_model(model, optimizer, scheduler, epoch, modelname):
    model_cpu = model.to('cpu')
    state = {
            'model' : model_cpu.state_dict(),
            'optimizer':optimizer.state_dict(),
            'scheduler' : scheduler.state_dict()
            }
    path = '/mnt/data/guest0/uchan/saved_model'
    if not (os.path.isdir(path)) : os.mkdir(path)
    torch.save(state, os.path.join(path,'{}.pth'.format(modelname)))


def main():
    print(torch.__version__)
    cudnn.benchmark = False
    device = torch.device('cuda:{}'.format(params['gpu'][0]))
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    logdir = os.path.join(params['log'], cur_time)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    writer = SummaryWriter(log_dir=logdir)
    #convert_csv(params['dataset'])
    print('Loading dataset')
    train_transform = create_train_transform(True,True,False,True, size=params['size'], bright=False)
    train_dataset = VideoDataset(params['dataset'],size=params['size'], mode='train', play_time=params['clip_len'],frame_sample_rate=params['frame_sample_rate'], transform=train_transform, sub_classnum = params['num_classes'], label_num = params['label_num'], stride_num = params['stride'])
    train_dataloader = DataLoader(
                        train_dataset,
                        batch_size=params['batch_size'],
                        shuffle=True,
                        num_workers=params['num_workers'])

    val_transform = create_val_transform(True,size=params['size'])
    val_dataset = VideoDataset(params['dataset'], size=params['size'],mode='validation', play_time=params['clip_len'],frame_sample_rate=params['frame_sample_rate'], transform = val_transform, sub_classnum = params['num_classes'], label_num=params['label_num'], stride_num = params['stride'])
    val_dataloader = DataLoader(
                        val_dataset,
                        batch_size=params['batch_size'],
                        shuffle=False,
                        num_workers=params['num_workers'])

    print('train_dataset : batch_size -> {}, step_size -> {}, frames -> {}'.format(params['batch_size'],len(train_dataloader), params['clip_len']))
    print('validation_dataset : batch_size -> {}, step_size -> {}, frames -> {}'.format(params['batch_size'],len(val_dataloader), params['clip_len']))
    print('=========================================================================================================')
    print('Load model : mode -> {}, label_size -> {}, sub_class_num -> {}'.format(params['mode'], params['label_num'], params['num_classes']))
    
    ### regression ###
    
    #model = generate_model('XL', n_classes = params['label_num'])
    
    #### class - one label ###
    model = None
    if params['mode']=='single':
        if params['model']=='slowfast':
            model = slowfastnet.resnet50(class_num=params['num_classes'], label_num = params['label_num'], mode = params['mode'])
        elif params['model']=='x3d':
            model = x3d.generate_model('S', n_classes = params['num_classes'])

    ### multi ###
    elif params['mode']=='multi':
        
        if params['model']=='slowfast':
            model = multi_slowfastnet.resnet50(class_num=params['num_classes'], label_num = params['label_num'], mode = params['mode']) 
        elif params['model']=='x3d':
            model = multi_x3d.generate_model('S', n_classes = params['num_classes'], label_num = params['label_num'])
        
        elif params['model'] =='eff':
            model = efficientnet.EfficientNet3D.from_name('efficientnet-b{}'.format(params['eff']), override_params={'num_classes': params['num_classes']}, mode = params['mode'], label_num = params['label_num'])
        

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
    '''
    criterion = Custom_MSELoss(num_classes = params['num_classes']).cuda()
    '''
    ### classification ###
    if params['mode']=='single':
        '''
        criterion = nn.CrossEntropyLoss(weight = train_dataset.get_class_weight().to(device))
        '''
        criterion =  Custom_CrossEntropyLoss(weight = train_dataset.get_class_weight().to(device))
    elif params['mode']=='multi':

        ### multi-class ##
        
        criterion = Custom_MultiCrossEntropyLoss(weight = train_dataset.get_class_weight().to(device), label_num=params['label_num'])
    
    #optimizer = optim.SGD(model.parameters(), lr = params['learning_rate'], momentum = params['momentum'], weight_decay = params['weight_decay'])
    #scheduler = optim.lr_scheduler.StepLR(optimizer,  step_size = params['step'], gamma=0.1)

    #optimizer = optim.SGD(model.parameters(),lr = params['learning_rate'],weight_decay=params['weight_decay'])

    optimizer = optim.SGD([
        {'params': list(model.parameters())[:-8]},
        {'params': model.fcs.parameters(), 'lr': params['learning_rate']}]
        , lr = params['learning_rate']*0.1, weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience = 4, factor = 0.1, verbose=True)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 30, eta_min = 0)
    model_save_dir = os.path.join(params['save_path'], 'second')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    print("train gogosing")
    pre_metric = 0
    for epoch in range(params['epoch_num']):
        train(model, train_dataloader, epoch, criterion, optimizer, writer, device, mode = params['mode'], label_num=params['label_num'], display = params['display'])
        if (epoch+1) % 5==0:
            print('======================================================')
            print('validation gogosing')
            metric = val(model, val_dataloader, epoch, criterion, optimizer, writer, device, mode=params['mode'], label_num=params['label_num'])
            #validation_loss, metric = val(model, val_dataloader, epoch, criterion, optimizer, writer, device, mode=params['mode'], label_num=params['label_num'])
            #print('validation loss -> {loss:.5f}, \t f1_score -> {f1_score:.5f}'.format(loss = validation_loss, f1_score = metric))
            #checkpoint = os.path.join(model_save_dir, str(epoch) + '.pth.tar')
            #torch.save(model.state_dict(),checkpoint)
            if metric>pre_metric:
                pre_metric = metric
                if params['model'] == 'eff':
                    save_model(model, optimizer, schediler, epoch, params['model'] + params['eff'])
                else:
                    save_model(model,optimizer,scheduler,epoch, params['model'])
                model = model.to(device)
            print('Total F1_score : {metrics:.5f}'.format(metrics = metric))
            print('======================================================')

            scheduler.step(metric)
        #scheduler.step()

    writer.close()


if __name__=='__main__':
    main()
