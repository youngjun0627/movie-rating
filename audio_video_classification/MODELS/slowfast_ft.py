'''
Here's the snippet from my code.
Also modify the .yaml config file as needed.
Hope this helps! 
'''
import torch
import numpy as np
from .slowfast.models.build import build_model
from .slowfast.models import head_helper
from .slowfast.config.defaults import get_cfg
from .slowfast.models import checkpoint as cu 
import os

def get_slowfast(device):
    # slowfast net
    cfg = get_cfg()
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Load config from cfg.
    root = '/home/uchanlee/uchanlee/uchan/video_classification/MODELS/slowfast'
    cfg_file = os.path.join(root,'config/slowfast_config.yaml')
    cfg.merge_from_file(cfg_file)
    slowfast_model = build_model(cfg, gpu_id=device)
    cfg.TRAIN.CHECKPOINT_TYPE = 'caffe2'
    cfg.TRAIN.CHECKPOINT_FILE_PATH = os.path.join(root,'saved_model/SLOWFAST_8x8_R50.pkl')
    #slowfast_model.load_state_dict(torch.load(cfg.TRAIN.CHECKPOINT_FILE_PATH), strict=False, encoding='latin1')
     
    cu.load_checkpoint(
    cfg.TRAIN.CHECKPOINT_FILE_PATH,
      slowfast_model,
        False,
          None,
            inflation=False,
              convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
              )
    
    '''
    inputs = torch.randn((2,3,1024,112,112)).to(device)
    content, genre, age = slowfast_model(inputs)

    print(content.shape)
    print(genre.shape)
    print(age.shape)
    '''
    return slowfast_model

if __name__=='__main__':
    get_slowfast(torch.device('cuda:1'))
