import os
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
import csv
import albumentations
from TRANSFORMS.transform import create_train_transform
from albumentations.pytorch.functional import img_to_tensor
import torch
import random

class VideoDataset(Dataset):
    def __init__(self, directory,size=224, mode = 'train', frame_sample_rate=1, cut_time=1, play_time=680,transform=None,sub_classnum=4, label_num = 1):
        folder = directory
        self.size = size
        self.frame_sample_rate = frame_sample_rate
        self.mode = mode
        self.play_time = play_time
        #self.frame_indices = range(start_time, start_time+play_time)
        self.cut_time = cut_time
        self.transform = transform
        self.filenames = []
        self.labels = []
        self.frame_lengths = []
        self.sub_classnum = sub_classnum
        self.label_num = label_num

        if mode == 'train':
            csv_file = os.path.join(folder, 'train-for_user.csv')
        elif mode == 'validation' :
            csv_file = os.path.join(folder, 'val-for_user.csv')
        else:
            raise Exception('{} is Unsupported'.format(mode))

        with open(csv_file, 'r', encoding='utf-8-sig') as f:
            rdr = csv.reader(f)
            for idx, data in enumerate(rdr):
                frame_length =int(data[0])
                filename = data[1]
                if self.label_num==1:
                    label = data[3]
                    label = self.convert_label(label)
                    self.labels.append(label)
   
                else: 
                    labels = []      
                    for label in data[2:7]:
                        label = self.convert_label(label)
                        labels.append(label)
                    self.labels.append(labels)
                    

                self.filenames.append(filename)
                self.frame_lengths.append(frame_length)
                #self.labels.append(labels)
        dic = {}
        if label_num==1:
            for _label in self.labels:
                if _label not in dic:
                    dic[_label]=0
                dic[_label]+=1
            print('mode : {} -> {}'.format(self.mode, dic))
        else:
            for _labels in self.labels:
                for _label in _labels:
                    if _label not in dic:
                        dic[_label]=0
                    dic[_label]+=1
            print('mode : {} -> {}'.format(self.mode, dic))
        self.class_weight = dic

    def __getitem__(self,index):
        video_path = self.filenames[index]
        frame_length = self.frame_lengths[index]
        if self.mode=='train':
            start = random.randrange(self.cut_time, frame_length-self.play_time-10)
        elif self.mode=='validation':
            start = self.cut_time
        frame_indices = range(start,start+self.play_time)
        video = self.load_clip_video(video_path, frame_indices)
        label = self.labels[index]
        return video, label

    def __len__(self):
        return len(self.filenames)

    def load_clip_video(self, video_path, frame_indices):
        video = np.empty((self.play_time//8, self.size, self.size, 3), np.dtype('float32')) ## make
        idx=0
        seed = random.randint(0,99999)
        for i in frame_indices:
            if i%8!=0: ## make
                continue
            random.seed(seed)
            image_path = os.path.join(video_path, 'frame_{:05d}.jpg'.format(i))
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(image = img)['image']
            #img = self.normalize(img)
            video[idx]=img
            idx+=1
        output = self.to_tensor(video)
        return output
    
    def load_image(image_path):
        with open(image_path, 'rb') as f:
            with image.open(f) as img:
                return img.convert('RGB')

    def convert_label(self, label):
        if self.sub_classnum==4:
            dic = {0:0,1:1,2:2,3:3}
        elif self.sub_classnum==3:
            dic = {0:0,1:0,2:1,3:1,4:2,5:2}
        elif self.sub_classnum==2:
            dic = {0:0,1:0,2:1,3:1}

        return dic[int(label)]

    def get_class_weight(self):
        class_num = len(self.class_weight.keys())
        weight = [0 for _ in range(class_num)]
        for k,v in self.class_weight.items():
            weight[k]=v
        weight = torch.tensor(weight)
        weight = weight / weight.sum()
        weight = 1.0 / weight
        weight = weight / weight.sum()
        return weight



    def normalize(self, img):
        return img

    def to_tensor(self, _input):
        return _input.transpose((3,0,1,2))

#if __name__=='__main__':
    

    #transform = create_train_transform(True,True,True,True,size=112)
    #path = '/home/guest0/uchan/slowfast/UTILS'
    #a = VideoDataset(path, transform = transform, size=112,label_num=4)
    #print((a[0][0]>0).sum())
    #print((a[0][0]<0).sum())
    '''
    r_mean = 0
    r_std = 0
    g_mean=0
    g_std=0
    b_mean=0
    b_std=0

    for b in a:
        data = b[0]
        r_mean += (data[0,:,:,:]/255.0).mean()
        r_std += (data[0,:,:,:]/255.0).std()
        g_mean += (data[1,:,:,:]/255.0).mean()
        g_std += (data[1,:,:,:]/255.0).std()
        b_mean += (data[2,:,:,:]/255.0).mean()
        b_std += (data[2,:,:,:]/255.0).std()
    print(r_mean/len(a), r_std/len(a))
    print(g_mean/len(a), g_std/len(a))
    print(b_mean/len(a), b_std/len(a))
    '''
