
params = dict()
params['num_classes']=4
params['dataset'] = '/home/uchanlee/uchanlee/uchan/final_project2/UTILS'
params['use_plot'] = True
params['epoch_num']=300
params['label_num'] = 1
params['mode']='multi'
params['batch_size']=16
params['step']=50
params['model'] = 'slowfast_multitask'
params['size']=112
params['num_workers']=4
params['momentum']=0.9
params['learning_rate']=0.001
params['weight_decay']=1e-6
params['display']=50
params['clip_len']=1024
params['stride'] = 4
params['pretrained']=''#'saved_model/saved_model_50.pth'
params['log']='log'
params['gpu']=[1]
params['save_path'] = 'saved'
params['frame_sample_rate']=1
params['patience']=2
params['K_fold']=True
