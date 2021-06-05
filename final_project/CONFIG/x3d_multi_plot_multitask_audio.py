
params = dict()
params['num_classes']=4
params['dataset'] = '/home/uchanlee/uchanlee/uchan/slowfast_multitask_audio/UTILS'
params['use_plot'] = True
params['epoch_num']=300
params['label_num'] = 5
params['mode']='multi'
params['batch_size']=2
params['step']=50
params['model'] = 'x3d_multitask'
params['size']=112
params['num_workers']=4
params['momentum']=0.9
params['learning_rate']=0.015
params['weight_decay']=1e-5
params['display']=400
params['clip_len']=1024
params['stride'] = 1
params['pretrained']=''#'saved_model/saved_model_50.pth'
params['log']='log'
params['gpu']=[1]
params['save_path'] = 'saved'
params['frame_sample_rate']=1
params['patience']=5
params['K_fold']=True
