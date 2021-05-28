import csv
import os
import shutil
def convert_csv(path):
    topics = ['moviename',
            'Positive Messages',
            'Positiva Role Models & Representations',
            'Violence',
            'Sex',
            'Language',
            'Consumerism',
            'Drinking, Drugs & Smoking',
            'Educational Value',
            'Violence & Scariness',
            'Sexy Stuff']

    print(path , ' : ', os.path.exists(path))
    #train_f = open(os.path.join(path, 'train-for_user.csv'), 'w', encoding = 'utf-8-sig', newline='')
    #val_f = open(os.path.join(path, 'val-for_user.csv'), 'w', encoding = 'utf-8-sig', newline='')
    train_f = open('train-for_user.csv', 'w', encoding = 'utf-8-sig', newline='')
    val_f = open('val-for_user.csv', 'w', encoding = 'utf-8-sig', newline='')
    train_wr = csv.writer(train_f)
    val_wr = csv.writer(val_f)
    cnt=0
    for moviename in os.listdir(path):
        cnt+=1
        if moviename.endswith('.csv'):
            continue
        videopath = os.path.join(path, moviename)
        data_f = open(os.path.join(videopath, 'data.csv'), 'r', encoding='utf-8-sig')
        frames_length = len(os.listdir(videopath))-1
        rdr = csv.reader(data_f)
        temp_dic = {}
        names = []
        for idx, value in enumerate(rdr):
            if idx==0:
                for v in value:
                    temp_dic[v]=0
                    names.append(v)
            if idx==2:
                for name , num in zip(names, value):
                    temp_dic[name]=num
                if len(temp_dic) !=9:
                    for topic in topics[1:]:
                        if topic not in temp_dic:
                            temp_dic[topic]=0

        data_f.close()
        if cnt%10!=0:
            train_wr.writerow([frames_length,videopath,
                str((int(temp_dic[topics[1]])+int(temp_dic[topics[2]]))//2),
                temp_dic[topics[3]],
                temp_dic[topics[4]],
                temp_dic[topics[5]],
                temp_dic[topics[7]],
                temp_dic[topics[8]],
                temp_dic[topics[9]]])
        else:
            val_wr.writerow([frames_length,videopath,
                str((int(temp_dic[topics[1]])+int(temp_dic[topics[2]]))//2),
                temp_dic[topics[3]],
                temp_dic[topics[4]],
                temp_dic[topics[5]],
                temp_dic[topics[7]],
                temp_dic[topics[8]],
                temp_dic[topics[9]]])

    train_f.close()
    val_f.close()


def convert2_csv(path,k):

    topics = ['sex_nudity','violence_gore','profianity','alcohol_drugs_smoking','frightening_intense_scene','plot']
    print(path , ' : ', os.path.exists(path))
    #train_f = open(os.path.join(path, 'train-for_user.csv'), 'w', encoding = 'utf-8-sig', newline='')
    #val_f = open(os.path.join(path, 'val-for_user.csv'), 'w', encoding = 'utf-8-sig', newline='')
    train_f = open('train-for_user.csv', 'w', encoding = 'utf-8-sig', newline='')
    val_f = open('val-for_user.csv', 'w', encoding = 'utf-8-sig', newline='')
    train_wr = csv.writer(train_f)
    val_wr = csv.writer(val_f)
    cnt=0
    for moviename in os.listdir(path):
        cnt+=1
        if moviename.endswith('.csv'):
            continue
        videopath = os.path.join(path, moviename)
        data_f = open(os.path.join(videopath, 'data.csv'), 'r', encoding='utf-8-sig')
        frames_length = len(os.listdir(videopath))
        rdr = csv.reader(data_f)
        dic = {}
        names = []
        for idx, value in enumerate(rdr):
            if idx==0:
                for v in value:
                    dic[v]=None
            if idx==1:
                for i,meta in enumerate(value):
                    if i!=5:
                        if meta == 'None':
                            dic[topics[i]]=0
                        if meta == 'Mild':
                            dic[topics[i]]=1
                        if meta == 'Moderate':
                            dic[topics[i]]=2
                        if meta == 'Severe':
                            dic[topics[i]]=3
                    else:
                        dic[topics[i]]=meta

        data_f.close()
        #if (cnt+k)%5!=0:
        stand_length = 1024
        if frames_length>stand_length*1.2:
            rare_cnt=1
            
            if (dic[topics[4]]==3 or dic[topics[1]]==0) and (dic[topics[3]]!=1):
                #print(dic)
                #continue
                rare_cnt=3
                
            for _ in range(rare_cnt):
                train_wr.writerow([frames_length,videopath,
                    dic[topics[0]],
                    dic[topics[1]],
                    dic[topics[2]],
                    dic[topics[3]],
                    dic[topics[4]],
                    dic[topics[5]]])
        else:
            if frames_length>stand_length+5:
                val_wr.writerow([frames_length,videopath,
                    dic[topics[0]],
                    dic[topics[1]],
                    dic[topics[2]],
                    dic[topics[3]],
                    dic[topics[4]],
                    dic[topics[5]]])
    train_f.close()
    val_f.close()


def sample_video(video_path, FRAME_RATE, path_output):
    print(video_path)
    if video_path.endswith(('.mp4','.avi')):
        os.system('ffmpeg -ss 00:00:05 -i \"{0}\" -r \
                "{1}\" -q:v 2 -s 224x224 \"{2}/frame_%05d.jpg\"'.format(video_path, FRAME_RATE, path_output))
    else:
        raise ValueError('Video path is not the same of video file (.mp4 or .avi)')

def sampling(path):
    root_path = path
    videos_path = os.path.join(root_path,'movies')
    path_output = os.path.join(root_path,'DATA2')
    if not os.path.exists(path_output):
        os.mkdir(path_output)
    
    for videoname in os.listdir(videos_path):
        video_path = os.path.join(videos_path, videoname)
        if not os.path.exists(os.path.join(path_output, videoname)):
            os.mkdir(os.path.join(path_output, videoname))
        #print(os.path.exists(os.path.join(video_path,'video.mp4')))
        sample_video(os.path.join(video_path,'video.mp4'),16,os.path.join(path_output,videoname))
        #print(video_path,len(os.listdir(os.path.join(path_output,videoname))))
        shutil.copy(os.path.join(video_path,'data.csv'), os.path.join(path_output, videoname,'data.csv'))
        #print(os.path.exists(os.path.join(path_output, videoname,'frame_00001.jpg')))

if __name__=='__main__':
    #path = '/mnt/data/guest0/uchan'
    #sampling(path)
    path = '/mnt/data/guest0/uchan/DATA3'
    convert2_csv(path,0)
    #convert_csv(path)
    #sampling(path)
