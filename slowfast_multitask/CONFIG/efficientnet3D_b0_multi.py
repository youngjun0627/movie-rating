
params = dict()
params['num_classes']=4
params['dataset'] = '/home/guest0/uchan/slowfast/UTILS'
params['DATA'] = '/mnt/data/guest0/uchan/DATA'
params['epoch_num']=300
params['label_num'] = 4
params['mode']='multi'
params['size']=112
params['stride']=4
params['model'] = 'eff'
params['eff'] = 0
params['batch_size']=16
params['step']=50
params['num_workers']=4
params['momentum']=0.9
params['learning_rate']=0.1
params['weight_decay']=1e-5
params['display']=50
params['clip_len']=1024
params['pretrained']=''#'saved_model/saved_model_50.pth'
params['log']='log'
params['gpu']=[0]
params['save_path'] = 'saved'
params['frame_sample_rate']=1
params['patience']=5
params['K_fold']=True
