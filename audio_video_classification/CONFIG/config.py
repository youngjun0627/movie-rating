
params = dict()
params['num_classes']=4
params['dataset'] = '/home/uchanlee/uchanlee/uchan/text_classification/UTILS'
params['use_plot'] = False
params['use_audio'] = True
params['use_video'] = True
params['epoch_num']=1000
params['label_num'] = 4
params['mode']='multi'
params['batch_size']=2
params['step']=50
params['model'] = 'slowfast_multitask'
params['size']=112
params['num_workers']=1
params['momentum']=0.9
params['learning_rate']=0.001
params['weight_decay']=1e-4
params['clip_len']=1024
params['stride'] = 1
params['pretrained']=''#'saved_model/slowfast_multitask.pth'#'saved_model/saved_model_50.pth'
params['log']='log'
params['gpu']=[1]
params['save_path'] = 'saved'
params['frame_sample_rate']=1
params['patience']=2
params['K_fold']=True
