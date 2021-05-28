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
        if size==112:
            translist+=[albumentations.Resize(128,128)]
        elif size==224:
            translist+=[albumentations.Resize(256,256)]
        translist+=[albumentations.RandomCrop(size,size,always_apply=True)]
    if flip:
        translist+=[albumentations.OneOf([
                albumentations.HorizontalFlip()],p=0.5)]

    if noise:
        translist+=[albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=5),
            albumentations.MedianBlur(blur_limit=5),
            albumentations.OpticalDistortion(),
            albumentations.GaussNoise(var_limit=(5.0,20.0))], p=0.65)]

    if bright:
        translist+=[albumentations.RandomBrightness(limit=0.2, always_apply=False)]

    if cutout:
        translist+=[albumentations.Cutout(max_h_size = int(size*0.1), max_w_size=int(size*0.1), num_holes=1,p=0.5)]

    translist+=[albumentations.Normalize(mean=(0.45, 0.45, 0.45), std = (0.225, 0.225, 0.225))]
    #translist+=[albumentations.Normalize(mean=(0.2481, 0.2292, 0.2131), std = (0.2167,0.2071,0.2014))]
    #trainlist+=[albumentations.Normalize(mean=(0.2539, 0.2348, 0.2189), std = (0.2195,0.2110,0.2061))]
    #translist+=[albumentations.Normalize(mean=(0.2580, 0.2360, 0.2215), std = (0.2235, 0.2132, 0.2100))]

    #translist+=[albumentations.Normalize(mean=(0.2527, 0.2343, 0.2177), std = (0.2171, 0.2082, 0.2026))]
    transform = albumentations.Compose(translist)
    return transform

def create_val_transform(resize,size=112):
    vallist = []
    if resize:
        vallist+=[albumentations.Resize(size,size)]

    vallist+=[albumentations.Normalize(mean=(0.45, 0.45, 0.45), std = (0.225, 0.225, 0.225))]
    #vallist+=[albumentations.Normalize(mean=(0.2580, 0.2360, 0.2215), std = (0.2235, 0.2132, 0.2100))]
    #vallist+=[albumentations.Normalize(mean=(0.2481, 0.2292, 0.2131), std = (0.2167,0.2071,0.2014))]
    #vallist+=[albumentations.Normalize(mean=(0.2597, 0.2405, 0.2231), std = (0.2276,0.2196,0.2160))]
    transform = albumentations.Compose(vallist)
    return transform

