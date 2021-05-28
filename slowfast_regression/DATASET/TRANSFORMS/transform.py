import os
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
import csv
from albumentations.pytorch.functional import img_to_tensor
import albumentations
import torch

def create_train_transform(flip,
        noise,
        cutout,
        resize,
        size = 112,
        bright = True):
    
    translist = []
    if resize:
        translist+=[albumentations.Resize(128,128)]
        translist+=[albumentations.RandomCrop(size,size,always_apply=True)]
    if flip:
        translist+=[albumentations.OneOf([
                albumentations.HorizontalFlip()],p=0.5)]

    if noise:
        translist+=[albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=5),
            albumentations.MedianBlur(blur_limit=5),
            albumentations.OpticalDistortion(),
            albumentations.GaussNoise(var_limit=(5.0,30.0))], p=0.5)]

    if bright:
        translist+=[albumentations.RandomBrightness(limit=0.2, always_apply=False)]

    if cutout:
        translist+=[albumentations.Cutout(max_h_size = int(size*0.2), max_w_size=int(size*0.2), num_holes=1,p=0.3)]

    transform = albumentations.Compose(translist)
    return transform

def create_val_transform(resize,size=112):
    vallist = []
    if resize:
        vallist+=[albumentations.Resize(size,size)]

    transform = albumentations.Compose(vallist)
    return transform

