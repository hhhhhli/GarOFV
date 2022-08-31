from time import time
from unittest import result
import torch
import torch.nn as nn
import numpy as np

from archs.solvers import extract_embeddings_test, Frame_optical, Frame_optical_HS
from archs.datasets import GarNet_Dataset, OF_Dataset
from torch.utils.data import DataLoader

import pandas as pd
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Any,Callable,Dict,IO,Optional,Tuple,Union
import os

import scipy.stats
import matplotlib.pyplot as plt
from shapely import geometry
import argparse
import csv


class AlexNet_Embedding(nn.Module):
    def __init__(self):
        super(AlexNet_Embedding,self).__init__()
        self.model=frozon(models.alexnet(pretrained=True))
        self.model.features[0]=nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        self.model.classifier=nn.Sequential(
            nn.Linear(256*6*6,256),
            nn.PReLU(),
            nn.Linear(256,256),
            nn.PReLU(),
            nn.Linear(256,2)
        )
    
    def forward(self,x):
        output=self.model(x)
        return output
    
    def get_emdding(self,x):
        return self.forward(x)

class ResNet18_Embedding(nn.Module):
    def __init__(self) -> None:
        super(ResNet18_Embedding,self).__init__()
        modeling=frozon(models.resnet18(pretrained=True))
        modules=list(modeling.children())[:-2]
        self.features=nn.Sequential(*modules)
        self.features[0]=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc=nn.Sequential(
            nn.Linear(512*8*8,256),
            nn.PReLU(),
            nn.Linear(256,256),
            nn.PReLU(),
            nn.Linear(256,2)
        )

    def forward(self,x):
        output=self.features(x)
        output=output.reshape(output.shape[0],-1)
        output=self.fc(output)
        return output
    
    def get_emdding(self,x):
        return self.forward(x)

class TripletNet(nn.Module):
    def __init__(self,embedding_net):
        super(TripletNet,self).__init__()
        self.embedding_net=embedding_net
    
    def forward(self,x1,x2,x3):
        output1=self.embedding_net(x1)
        output2=self.embedding_net(x2)
        output3=self.embedding_net(x3)
        return output1,output2,output3
    
    def get_emdding(self,x):
        return self.embedding_net(x)

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

# 获取contours和standard points
def get_contours(embeddings, labels, n = 5, bandwidth_value = 10):
    # bandwidth_value=10
    # frame_optical = Frame_optical(dataloader_OF, len_video)
    print('bd = ', bandwidth_value)


    contours=[]
    standard_points=np.zeros((n,2))

    if n==5:
        color=['#1f77b4','#ff7f01','#2ca02c','#d62728','#9467bd','#1f77b4','#ff7f01','#2ca02c','#d62728','#9467bd']
        label_plottings=['pant','shirt','sweater','towel','tshirt']
    if n==3:
        color=['#1f77b4','#ff7f01','#2ca02c','#1f77b4','#ff7f01','#2ca02c']
        label_plottings=['light','medium','heavy']
    
    # calculate
    for i in range(n):
        # +n是因为测试集的label在6-10
        # inds = np.where(labels==i+1+(n))[0]
        inds = np.where(labels==i+1)[0]
        # print(inds)
        # exit(123)
        data = embeddings[inds]*10
        x, y = data[:,0].mean(), data[:,1].mean()
        # pdf=scipy.stats.kde.gaussian_kde(data.T)
        pdf=scipy.stats.gaussian_kde(data.T)
        q,w=np.meshgrid(range(-40,60,1), range(-40,40,1))
        r=pdf([q.flatten(),w.flatten()])
        s=scipy.stats.scoreatpercentile(pdf(pdf.resample(1000)), bandwidth_value)
        r.shape=(80,100)
        cont=plt.contour(range(-40,60,1), range(-40,40,1), r, [s],colors=color[i])
        cont_location=[]
        for line in cont.collections[0].get_paths():
            cont_location.append(line.vertices)
        # cont_location=np.array(cont_location, dtype=object)[0]
        cont_location=np.array(cont_location)[0]
        contours.append(cont_location)
        plt.plot(x,y,'o',color=color[i],label=label_plottings[i])
        standard_points[i,:]=(x,y)
    # plt.show()
    return contours, standard_points

# 转换测试集索引
# 返回video_idx, category_idx
def transform_video_idx(video_idx, n = 5):
    if n==5:
        if video_idx+1<=10:
            video_idx=video_idx+41
            category_idx=0
        elif video_idx+1>10 and video_idx+1<=20:
            video_idx=video_idx+71
            category_idx=1
        elif video_idx+1>20 and video_idx+1<=30:
            video_idx=video_idx+101
            category_idx=2
        elif video_idx+1>30 and video_idx+1<=40:
            video_idx=video_idx+131
            category_idx=3
        elif video_idx+1>40:
            video_idx=video_idx+161
            category_idx=4
    if n==3:
        if video_idx+1<=10:
            video_idx=video_idx+41
            category_idx=2
        elif video_idx+1>10 and video_idx+1<=20:
            video_idx=video_idx+71
            category_idx=1
        elif video_idx+1>20 and video_idx+1<=30:
            video_idx=video_idx+101
            category_idx=2
        elif video_idx+1>30 and video_idx+1<=40:
            video_idx=video_idx+131
            category_idx=0
        elif video_idx+1>40:
            video_idx=video_idx+161
            category_idx=1
    return video_idx, category_idx


# 获取每帧的分类结果 video_data = embedding[inds]*10
# 返回frame_result, mean_value_points
def get_frame_result(video_data, contours, standard_points, n = 5):
    # frame_result = np.zeros((len(video_data)))
    frame_result = np.ones((len(video_data))) * 100
    # mean_value_points = []
    for idx in range(len(video_data)):
        x=video_data[:idx+1,0].mean()# all previously observed frames
        y=video_data[:idx+1,1].mean()
        mean_value_point=np.array([x,y])
        contains_acc=[]

        for m in range (n):
            line = geometry.LineString(contours[m])
            point = geometry.Point(x,y)
            polygon = geometry.Polygon(line)
            if polygon.contains(point):
                contains_acc.append(m)
        if len(contains_acc)==1:
            # if contains_acc[0]==category_idx:
            #     acc+=1
            frame_result[idx] = contains_acc[0]
        elif len(contains_acc) > 1:
            dists=np.zeros((len(contains_acc),2))
            for h in range(len(contains_acc)):
                standard_point=standard_points[contains_acc[h]]
                dis=np.sum(np.power(standard_point-mean_value_point,2))
                dists[h,0]=dis
                dists[h,1]=contains_acc[h]
            min_val=np.argmin(dists[:,0])
            frame_result[idx] = dists[min_val,1]
        

    return frame_result

# 主逻辑
def video_classify(dataloader, dataloader_OF, model, Early_stop = False, Frame_optical_flag = True, \
                txt_file = "Frame_continuous-perception.txt", train_no = 0, bandwidth = 10, \
                Early_stop_thresh = 0.8, n = 5, result_dir='./results/', frame_optical_from_csv = False):
    result_csv = result_dir + 'Early_' + str(Early_stop) + '_Optical_' + str(Frame_optical_flag) + '.csv'
    if n==5:
        color=['#1f77b4','#ff7f01','#2ca02c','#d62728','#9467bd','#1f77b4','#ff7f01','#2ca02c','#d62728','#9467bd']
        label_plottings=['pant','shirt','sweater','towel','tshirt']
    if n==3:
        color=['#1f77b4','#ff7f01','#2ca02c','#1f77b4','#ff7f01','#2ca02c']
        label_plottings=['light','medium','heavy']

    num_video=50
    len_video=60

    # 获取embedding
    embeddings, labels, video_labels=extract_embeddings_test(dataloader, model)

    # 获取contours
    # bandwidth = 60 # 1-bandwidth=55% actually, in paper
    contours, standard_points = get_contours(embeddings, labels, n, bandwidth)
    # print(contours)
    # exit(123)

    # 获取光流差值
    frame_optical = None
    category_optical_sum = None
    if Frame_optical_flag == True:
        if frame_optical_from_csv == True:
            frame_optical = np.loadtxt('../loocv_experiments/Depth_And_RGB/optical_flows_no_'+str(train_no+1).zfill(2)+'.csv', delimiter=',')
        else:
            frame_optical = Frame_optical_HS(dataloader_OF, len_video)
        # exit(123)
    
    correct_video = 0
    video_acc = 0  
    start_time = time()

    acc_video, total_video = np.zeros((n)), np.zeros((n))

    for video_idx in range(num_video):

        if Frame_optical_flag == True:
            category_optical_sum = np.zeros((n))# [0,0,0,0,0]   光流累加值
        else:
            category_frame_num = np.zeros((n))# 帧数累加值
            
        total_frame = 0

        if frame_optical is not None:
            frame_optical_video = frame_optical[video_idx * (len_video-1):(video_idx+1) * (len_video-1)]# [1*59:2*59]
            frame_optical_video_sum = frame_optical_video.sum()
        # 转换测试集索引
        video_idx, category_idx = transform_video_idx(video_idx, n)

        # 获取每帧的分类结果
        inds=np.where(video_labels==video_idx)[0]
        # print(inds)
        # exit(0)
        frame_result = get_frame_result(embeddings[inds] * 10, contours, standard_points, n)
        # print(frame_result)
        # exit(999)

        # 遍历每帧的预测结果
        # Frame_optical_flag = True模式下 当前帧与前一帧的预测结果相同时 将当前帧的光流值加入累计值
        # Frame_optical_flag = False模式下 当前帧的预测结果即是截至至当前帧视频的预测结果
        # print(frame_result)
        for idx in range(len(frame_result)):
            total_frame += 1
            
            # 使用光流累计值进行判断
            if Frame_optical_flag == True:
                if idx > 0 and frame_result[idx] == frame_result[idx-1] and frame_result[idx] < 100:
                    category_optical_sum[int(frame_result[idx])] += frame_optical_video[idx-1]
                # if np.argmax(category_optical_sum) == category_idx:
                #     category_frame_num[int(frame_result[idx])] += 1
                # print('frame: {}, category_optical_sum = {}'.format(idx, category_optical_sum/category_optical_sum.sum()))
                
            # 每帧累计分属5类的预测帧数
            else:
                if frame_result[idx] < 100:
                    category_frame_num[int(frame_result[idx])] += 1# e.g. [0,2,0,10,1]for frame 13 in video capturing 4th category
                # print('frame: {}, category_frame_num = {}'.format(idx, category_frame_num))
                
                            
            # Early_stop模式下，满足条件的话提前停止
            if Early_stop == True:
                if Frame_optical_flag == False:
                    if (idx+1)>=20 and np.max(category_frame_num) / (idx+1) >= Early_stop_thresh:# 任意一类达到thresh以上
                        break
                    # else:
                    #     # 我们算法的提前停止条件是什么？或者无
                else:
                    if (idx+1)>=20 and np.max(category_optical_sum) / (idx+1) >= Early_stop_thresh:
                        break

        stop_frame = len_video
        video_pred = None
        ##############   计算视频prediction ##############
        # 结束60帧的判断，现在得到了第60帧/停止时的 category_optical_sum, category_frame_num, 和idx+1(停止帧数)
        # 下面需要计算当前视频的判断结果， 再与category比较判断正确与否
        if Frame_optical_flag == True:
            video_pred = np.argmax(category_optical_sum)
        else:
            if Early_stop == True:
                if idx+1 < len(frame_result):
                    video_pred = np.argmax(category_frame_num)# 当前的视频判断结果
                    stop_frame = idx+1 # 1~60帧
                else:# 第60帧没有判断出结果。那恰好第60帧stop，和第60帧也没有结果，2个情况应该怎么区分？
                    video_pred = None
            else:
                video_pred = np.argmax(category_frame_num)# 当前的视频判断结果

        with open(os.path.join(result_dir, txt_file.replace('.txt', '_detail.txt')), 'a') as file1:
            if video_pred != None:
                print ('[train]: no.',str(train_no+1).zfill(2),',bandwidth,', 100-bandwidth, ',[video_idx],',video_idx,',[Category],',label_plottings[category_idx],',[Pred],',str(label_plottings[video_pred]),',[Time],', str(time()-start_time), file=file1)
            else:
                print ('[train]: no.',str(train_no+1).zfill(2),',bandwidth,', 100-bandwidth, ',[video_idx],',video_idx,',[Category],',label_plottings[category_idx],',[Pred],',"None",',[Time],', str(time()-start_time), file=file1)

        ##############   计算所有视频的判断acc ##############
        # 下面要判断这个视频的预测结果，与category比较是否相同。用于计算所有视频的acc
        if video_pred ==  category_idx:
            correct_video +=1
        total_video[category_idx] += 1
        if video_pred ==  category_idx:
            acc_video[video_pred] += 1

    # print(total_video, acc_video)
    # exit(123)

    video_acc = correct_video / num_video
    with open(os.path.join(result_dir, txt_file), 'a') as file0:
        print('[train]: no.',str(train_no+1).zfill(2), ',bandwidth,', 100-bandwidth, ',category accuracy, ', acc_video[0]/total_video[0],',', acc_video[1]/total_video[1],',',acc_video[2]/total_video[2],',',acc_video[3]/total_video[3],',',acc_video[4]/total_video[4], ',', acc_video.sum()/total_video.sum(), file=file0)
    print('video_acc = ', video_acc)
                 


# 测试
if __name__ == '__main__':
    pars=argparse.ArgumentParser()
    pars.add_argument('--bandwidth',type=int,default=25,help='bandwidth value')
    pars.add_argument('--use_flow',type=str,default='True',help='use optical flow or not')
    pars.add_argument('--use_early_stop',type=str,default='False',help='use early stop or continuous perception')
    par=pars.parse_args()

    mean,std=0.00586554,0.03234654
    batch_size = 32
    kwargs={'num_workers':4,'pin_memory':True} if torch.cuda.is_available() else {}
    # file_path='../loocv_experiments/Depth_And_RGB/Database/depth'
    # model_path='../loocv_experiments/Depth_And_RGB/Model/depth/shapes/'
    file_path='./Database/'
    model_path='./Models/'
    data='/'

    num_train=4
    #ci_shapes=[[-30,30,-30,20,50,60],[-30,30,-30,30,60,60],[-30,40,-30,30,60,70],[-50,20,-30,40,70,70]]
    ci_shapes=[[-30,50,-30,30,60,80],[-30,30,-30,40,70,60],[-20,30,-20,30,50,50],[-40,30,-50,20,70,70]]#depth

    for train_no in range(num_train):
        model=torch.load(os.path.join(model_path, 'train_no_'+str(train_no+1).zfill(2)+'.pth'))
        # model = None
        csv_path='./explore_file/no_'+str(train_no+1).zfill(2)+'/explore.csv'
        dataset=GarNet_Dataset(file_path+data,csv_path,transform=transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((mean,),(std,))
        ]),opt=1)
        dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=False,**kwargs)
        
        if 'depth' in file_path:
            file_path_OF = file_path.replace('depth', 'rgb')
        else:
            file_path_OF = file_path
        dataset_OF=OF_Dataset(file_path_OF+data,csv_path,transform=transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((mean,),(std,))]),opt=1)## 这里的mean, std可能需要再算一下
        dataloader_OF=DataLoader(dataset_OF,batch_size=1,shuffle=False,**kwargs)




        use_flow = par.use_flow
        use_early_stop = par.use_early_stop
        bandwidth = int(par.bandwidth)

        # print(type(use_flow), use_early_stop, bandwidth)

        if use_flow == 'True':
            if use_early_stop == 'True':
                txt_file = "Optical-flow_early-stop.txt"
                video_classify(dataloader, dataloader_OF, model, True, True, txt_file, train_no, bandwidth)
            else:
                txt_file = "Optical-flow_continuous-perception.txt"
                video_classify(dataloader, dataloader_OF, model, False, True, txt_file, train_no, bandwidth)
        else:
            if use_early_stop == 'True':
                txt_file = "Frame_early-stop.txt"
                video_classify(dataloader, dataloader_OF, model, True, False, txt_file, train_no, bandwidth)
            else:
                txt_file = "Frame_continuous-perception.txt"
                video_classify(dataloader, dataloader_OF, model, False, False, txt_file, train_no, bandwidth)