import os
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import csv
import albumentations
from TRANSFORMS.transform import create_train_transform, create_val_transform
from albumentations.pytorch.functional import img_to_tensor
from sklearn import preprocessing
import torch
import random
import sklearn
import librosa
from torchtext.vocab import Vocab
from collections import Counter
from torchtext.data.utils import get_tokenizer

class VideoDataset(Dataset):
    def __init__(self, directory,size=224, mode = 'train', frame_sample_rate=1, cut_time=30, play_time=1024,transform=None,sub_classnum=4, label_num = 1, stride_num=8, use_plot=True, use_audio=True):
        folder = directory
        self.size = size
        self.frame_sample_rate = frame_sample_rate
        self.mode = mode
        self.play_time = play_time
        self.stride_num = stride_num
        #self.frame_indices = range(start_time, start_time+play_time)
        self.cut_time = cut_time
        self.transform = transform
        self.filenames = []
        self.labels = []
        self.frame_lengths = []
        self.sub_classnum = sub_classnum
        self.label_num = label_num
        self.plots = []
        self.use_plot = use_plot
        self.vocab = None
        self.text_pipeline = None
        self.sub_labels = []
        self.audios = []
        self.pad2d = lambda a, i:a[:, 0:i] if a.shape[1]>i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))
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
                    label = data[6]
                    label = self.convert_label(label)
                    self.labels.append(label)
   
                else: 
                    labels = []     
                    for _idx, label in enumerate(data[2:7]):
                        label = self.convert_label(label)
                        labels.append(label)
                        
                    self.labels.append(labels)

                if use_plot:
                    self.plots.append(data[7])

                if use_audio:
                    self.audios.append(data[10])

                self.filenames.append(filename)
                self.frame_lengths.append(frame_length)
                #self.labels.append(labels)
                if self.mode=='train' or self.mode=='validation':
                    labels = []
                    for _idx, label in enumerate(data[8:10]):
                        if _idx==0: # rating
                            labels.append(int(label))
                        elif _idx == 1: # genre
                            label = list(map(int,label[1:-1].split(',')))
                            labels.append(label)
                    self.sub_labels.append(labels)

        if label_num==1:
            dic = {}
            for _label in self.labels:
                if _label not in dic:
                    dic[_label]=0
                dic[_label]+=1
            self.class_weight = dic
        else:
            label_names = {0:'sex', 1:'violence', 2:'profinancy', 3: 'drug_smoking', 4: 'frighten'}
            #dic = {0:{}, 1: {}, 2:{}, 3:{}, 4:{}}
            #label_names = {0:'sex', 1:'violence', 2:'profinancy', 3: 'frighten'}
            dic = {0:{}, 1: {}, 2:{}, 3:{}, 4:{}}
            for _labels in self.labels:
                for i, _label in enumerate(_labels):
                    if _label not in dic[i]:
                        dic[i][_label]=0
                    dic[i][_label]+=1
            self.class_weight = dic
        print(self.class_weight)

    def __getitem__(self,index):
        video_path = self.filenames[index]
        frame_length = self.frame_lengths[index]

        if self.mode=='train':
            frame_indices = np.linspace(self.cut_time, frame_length-20, self.play_time+1)
            frame_indices = list(map(int, frame_indices))
            candidate = frame_indices[:]
            frame_indices = []
            for i in range(self.play_time):
                frame_indices.append(random.randrange(candidate[i], candidate[i+1]))
        elif self.mode=='validation':
            frame_indices = np.linspace(self.cut_time, frame_length-20, self.play_time)
            frame_indices = list(map(int, frame_indices))
        video = self.load_clip_video(video_path, frame_indices)
        label = self.labels[index]
        #print(type(video), type(label))

        if self.use_plot:# add audio 
            plot = self.plots[index]
            plot = np.array(self.text_pipeline(plot), dtype = np.long)
            #sig, sr = librosa.core.load(self.audios[index], 16000)

            #mfccs = librosa.feature.mfcc(y=sig, sr=sr, hop_length=160, n_mfcc=40, n_fft=2048)
            #mfccs = cv2.imread(self.audios[index], cv2.IMREAD_GRAYSCALE)
            #mfccs = np.load(self.audios[index])
            #_, audio_length = mfccs.shape
            #audio_start = audio_length * start // frame_length
            #audio_end = (audio_length * (start + self.play_time))// frame_length
            #mfccs = mfccs[:,frame_indices]
            #mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
            #mfccs = self.pad2d(mfccs, 4400)
            #mfccs = preprocessing.scale(np.array(mfccs).astype('float'))
            mfccs = np.load(self.audios[index]).astype('uint8')
            mfccs = Image.fromarray(mfccs).resize((4400,40))
            mfccs = np.array(mfccs).astype('float')
            if self.mode=='train':
                mfccs += 0.005 * np.random.randn(40,4400)
            mfccs = np.expand_dims(mfccs, axis=0)
            return video, plot, np.array(label), self.sub_labels[index][0], np.array(self.sub_labels[index][1]), mfccs
        else:
            #return video, np.array(label), np.array([self.sub_labels[index][0]]), np.array(self.sub_labels[index][1])
            return video, np.array(label)

    def __len__(self):
        return len(self.filenames)

    def load_clip_video(self, video_path, frame_indices):
        video = np.empty((self.play_time//self.stride_num, self.size, self.size, 3), np.dtype('float32')) ## make
        seed = random.randint(0,99999)
        for idx, i in enumerate(frame_indices):
            if idx%self.stride_num!=0: ## make
                continue
            random.seed(seed)
            image_path = os.path.join(video_path, 'frame_{:05d}.jpg'.format(i))
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if self.transform:
                img = self.transform(image = img)['image']
            #img = self.normalize(img)
            video[idx//self.stride_num]=img
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

    def generate_text_pipeline(self, vocab, tokenizer):
        self.vocab = vocab
        self.text_pipeline = lambda x : [vocab[token] for token in tokenizer(x)]

    def get_class_weight(self):
        if self.label_num==1:
            class_num = len(self.class_weight.keys())
            weight = [0 for _ in range(class_num)]
            for k,v in self.class_weight.items():
                weight[k]=v
            weight = torch.tensor(weight)
            weight = weight / weight.sum()
            weight = 1.0 / weight
            weight = weight / weight.sum()
            return weight
        elif self.label_num==5:
            weights = []
            for i in range(self.label_num):
                weights.append([0 for _ in range(self.sub_classnum)])
                weight = self.class_weight[i]
                for k,v in weight.items():
                    weights[i][k]=v
                weights[i] = torch.tensor(weights[i])
                weights[i] = weights[i] / weights[i].sum()
                weights[i] = 1.0/ weights[i]
                weights[i] = weights[i] / weights[i].sum()
            return torch.stack(weights)
    
    def get_age_weight(self):
        dic={}
        for label in self.sub_labels:
            label = label[0]
            if label not in dic:
                dic[label]=0
            dic[label]+=1
        weights = [0 for _ in range(len(dic.keys()))]
        for k,v in dic.items():
            weights[k]=v
        weights = torch.tensor(weights)
        weights = weights / weights.sum()
        weights = 1.0 / weights
        weights = weights / weights.sum()

        return weights

    def get_class_weight2(self):
        if self.label_num==1:
            weights = [0 for _ in range(self.sub_classnum)]
            for k,v in self.class_weight.items():
                weights[k]=v
            weights = torch.tensor(weights)
            weights = torch.tensor([sum(weights)/(x*self.sub_classnum) for x in weights])
            return weights

        elif self.label_num==5 or self.label_num==4:
            weights = []
            for i in range(self.label_num):
                weights.append([0 for _ in range(self.sub_classnum)])
                weight = self.class_weight[i]
                for k,v in weight.items():
                    weights[i][k]=v
                weights[i] = torch.tensor(weights[i])
                weights[i] = torch.tensor([sum(weights[i])/(self.sub_classnum*x) for x in weights[i]])
            return torch.stack(weights)
    
    def get_age_weight2(self):
        dic={}
        for label in self.sub_labels:
            label = label[0]
            if label not in dic:
                dic[label]=0
            dic[label]+=1
        weights = [0 for _ in range(len(dic.keys()))]
        for k,v in dic.items():
            weights[k]=v
        weights = torch.tensor(weights)
        weights = torch.tensor([sum(weights)/(x*self.sub_classnum) for x in weights])
        return weights

    def normalize(self, img):
        return img/255.

    def to_tensor(self, _input):
        return _input.transpose((3,0,1,2))


if __name__=='__main__':
    

    
    transform = create_train_transform(True,True,True,True,size=112)
    path = '/home/uchanlee/uchanlee/uchan/final_project/UTILS'
    a = VideoDataset(path, transform = None, size=224,label_num=1, use_plot=False, mode='validation')
    
    plots = a.plots
    counter = Counter()
    tokenizer = get_tokenizer('basic_english')
    for plot in plots:
        counter.update(tokenizer(plot))
    vocab = Vocab(counter,min_freq=1)
    a.generate_text_pipeline(vocab,tokenizer)
    print(a.get_class_weight())
    '''
    r_mean = 0
    r_std = 0
    g_mean = 0
    g_std = 0
    b_mean = 0
    b_std = 0
    dataloader = DataLoader(a, batch_size=1)
    for video, _ in dataloader:
        r_mean += (video[:,0,:,:,:]/255.).mean()
        r_std += (video[:,0,:,:,:]/255.).std()
        g_mean += (video[:,1,:,:,:]/255.).mean()
        g_std += (video[:,1,:,:,:]/255.).std()
        b_mean += (video[:,2,:,:,:]/255.).mean()
        b_std += (video[:,2,:,:,:]/255.).std()
    print(r_mean/len(dataloader), g_mean/len(dataloader), b_mean/len(dataloader))
    print(r_std/len(dataloader), g_std/len(dataloader), b_std/len(dataloader))
    '''
        
    #print(a.get_class_weight())
    #print(a.get_class_weight2())
    #print(a.get_age_weight2())
    #print(a.get_age_weight())
    #for i in range(len(a.filenames)):
    #    print(i)
    '''
    transform = create_val_transform(True, size=112)
    a = VideoDataset(path, transform = transform, size=112,label_num=1, use_plot=True, mode = 'validation')
    
    plots = a.plots
    counter = Counter()
    tokenizer = get_tokenizer('basic_english')
    for plot in plots:
        counter.update(tokenizer(plot))
    vocab = Vocab(counter,min_freq=1)
    a.generate_text_pipeline(vocab,tokenizer)
    print(a.get_class_weight2())
    print(a.get_age_weight2())
    '''
    #print(a[0][0].shape)
    #for i in range(len(a.filenames)):
    #    print(i)
    '''
    r_mean=0
    r_std=0
    b_mean=0
    b_std=0
    g_mean=0
    g_std=0
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



    a = VideoDataset(path, transform = transform, size=224,label_num=5, use_plot=True, mode='validation')
    
    plots = a.plots
    counter = Counter()
    tokenizer = get_tokenizer('basic_english')
    for plot in plots:
        counter.update(tokenizer(plot))
    vocab = Vocab(counter,min_freq=1)
    a.generate_text_pipeline(vocab,tokenizer)
    print(a.get_class_weight())
    print(a.get_class_weight2())
    print(a.get_age_weight())
    r_mean=0
    r_std=0
    b_mean=0
    b_std=0
    g_mean=0
    g_std=0
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
    '''
    print(a[100][0].shape)
    b = torch.tensor(a[0][0])
    images = b.permute(1,2,3,0).numpy()
    for idx,img in enumerate(images):
        if idx ==1:
            print(img)
        cv2.imwrite('./TEMP/image_{}.jpg'.format(idx), img.astype('uint8'))
    '''
    '''
    video, plot, label, sub_labels, sub_labels2, mfccs = a[0]
    print(mfccs.shape)
    '''
    #print(a.get_class_weight())
    #print(len((a.sub_labels)))
    #print(a[0])
    #transform = create_val_transform(True, size=112)
    #a = VideoDataset(path,size=112, transform = transform, mode = 'validation', label_num=4)
    #print(a.get_class_weight())
    
    #print((a[0][0]>0).sum())
    #print((a[0][0]<0).sum())
    
    #r_mean = 0
    #r_std = 0
    #g_mean=0
    #g_std=0
    #b_mean=0
    #b_std=0
    
    #path = '/home/guest0/uchan/slowfast/UTILS'
    #transform = create_train_transform(True,True,True,True,size=112)
    #a = VideoDataset(path, transform = transform, size=112,label_num=4)
    #print(len(a))
    #print(a.get_class_weight())
    #transform = create_val_transform(True,size=112)
    #a = VideoDataset(path, transform = transform, size=112,label_num=4, mode = 'validation')
    #print(len(a))
    #print(a.get_class_weight())
    
    
    #path = '/home/guest0/uchan/slowfast/UTILS'
    #transform = create_train_transform(False,False,False,True,size=112)
    #a = VideoDataset(path, transform = transform, size=112,label_num=4)
    #transform = create_train_transform(True,size=112)
    #transform = create_train_transform(True,True,True,True,size=112)
    #a = VideoDataset(path, transform = transform, size=112,label_num=4)
    #r_mean=0
    #r_std=0
    #b_mean=0
    #b_std=0
    #g_mean=0
    #g_std=0
    #for b in a:
    #    data = b[0]
    #    r_mean += (data[0,:,:,:]/255.0).mean()
    #    r_std += (data[0,:,:,:]/255.0).std()
    #    g_mean += (data[1,:,:,:]/255.0).mean()
    #    g_std += (data[1,:,:,:]/255.0).std()
    #    b_mean += (data[2,:,:,:]/255.0).mean()
    #    b_std += (data[2,:,:,:]/255.0).std()
    #print(a[0][0])
    #print(r_mean/len(a), r_std/len(a))
    #print(g_mean/len(a), g_std/len(a))
    #print(b_mean/len(a), b_std/len(a))
    
