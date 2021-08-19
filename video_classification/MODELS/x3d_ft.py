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

def get_x3d(device):
    # slowfast net
    cfg = get_cfg()
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Load config from cfg.
    root = '/home/uchanlee/uchanlee/uchan/video_classification/MODELS/slowfast'
    cfg_file = os.path.join(root,'config/x3d_config.yaml')
    cfg.merge_from_file(cfg_file)
    model = build_model(cfg, gpu_id=device)
    cfg.TRAIN.CHECKPOINT_TYPE = 'caffe2'
    cfg.TRAIN.CHECKPOINT_FILE_PATH = os.path.join(root,'saved_model/x3d_m.pyth')
    #slowfast_model.load_state_dict(torch.load(cfg.TRAIN.CHECKPOINT_FILE_PATH), strict=False, encoding='latin1')
     
    cu.load_checkpoint(
    cfg.TRAIN.CHECKPOINT_FILE_PATH,
      model,
        False,
          None,
            inflation=False,
              #convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
              )
    '''
    inputs = torch.randn((2,3,1024,112,112)).to(device)
    content, genre, age = model(inputs)

    print(content.shape)
    print(genre.shape)
    print(age.shape)
    '''
    return model

if __name__=='__main__':
    get_x3d(torch.device('cuda:2'))
