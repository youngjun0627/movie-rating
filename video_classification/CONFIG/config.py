
params = dict()
params['num_classes']=4
params['dataset'] = '/home/uchanlee/uchanlee/uchan/final_project3/UTILS'
params['use_plot'] = False
params['use_audio'] = False
params['use_video'] = True
params['epoch_num']=300
params['label_num'] = 4
params['mode']='multi'
params['batch_size']=4
params['step']=50
params['model'] = 'slowfast_multitask'
params['size']=112
params['num_workers']=2
params['momentum']=0.9
params['learning_rate']=0.01
params['weight_decay']=1e-5
params['display']=740
params['clip_len']=1024
params['stride'] = 1
params['pretrained']=''#'saved_model/saved_model_50.pth'
params['log']='log'
params['gpu']=[1]
params['save_path'] = 'saved'
params['frame_sample_rate']=1
params['patience']=2
params['K_fold']=True
