import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from typing import Any,Callable,Dict,IO,Optional,Tuple,Union
import pandas as pd
import cv2
from PIL import Image

class PhySNet_Dataset(Dataset):
    def __init__(self,train:bool=True,transform:Optional[Callable]=None,target_transform:Optional[Callable]=None,opt=1,par=None)->None:
        super(PhySNet_Dataset,self).__init__()
        self.train=train
        self.train_file=par.train_file
        self.test_file=par.test_file
        self.img_path=par.img_path
        self.csv_path=par.csv_path
        if self.train:
            data_file=self.train_file
            data=pd.read_csv(data_file+self.csv_path)
            self.train_data=data.iloc[:,0]
            self.train_labels=data.iloc[:,opt]
        else:
            data_file=self.test_file
            data=pd.read_csv(data_file+self.csv_path)
            self.test_data=data.iloc[:,0]
            self.test_labels=data.iloc[:,opt]
        self.transform=transform
        self.target_transform=target_transform
    
    def __getitem__(self,index:int)->Tuple[Any,Any]:
        if self.train:
            imgs_path=self.train_file+self.img_path+self.train_data[index]
            target=int(self.train_labels[index])
        else:
            imgs_path=self.test_file+self.img_path+self.test_data.iloc[index]
            target=int(self.test_labels.iloc[index])
        img=cv2.imread(imgs_path,0)
        img=Image.fromarray(img,mode='L')
        if self.transform is not None:
            img=self.transform(img)
        if self.target_transform is not None:
            target=self.target_transform(target)
        noise=0.01*torch.rand_like(img)
        img=img+noise
        return img, target
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
    
class TripletMNIST(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform
        self.train_file=self.mnist_dataset.train_file
        self.test_file=self.mnist_dataset.test_file
        self.img_path=self.mnist_dataset.img_path

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.to_numpy())
            self.label_to_indices = {label: np.where(self.train_labels.to_numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.to_numpy())
            self.label_to_indices = {label: np.where(self.test_labels.to_numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = cv2.imread(self.train_file+self.img_path+self.train_data[index],0), self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = cv2.imread(self.train_file+self.img_path+self.train_data[positive_index],0)
            img3 = cv2.imread(self.train_file+self.img_path+self.train_data[negative_index],0)
        else:
            img1 = cv2.imread(self.test_file+self.img_path+self.test_data[self.test_triplets[index][0]],0)
            img2 = cv2.imread(self.test_file+self.img_path+self.test_data[self.test_triplets[index][1]],0)
            img3 = cv2.imread(self.test_file+self.img_path+self.test_data[self.test_triplets[index][2]],0)

        img1 = Image.fromarray(img1, mode='L')
        img2 = Image.fromarray(img2, mode='L')
        img3 = Image.fromarray(img3, mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        img1=img1+0.01*torch.rand_like(img1)
        img2=img2+0.01*torch.rand_like(img2)
        img3=img3+0.01*torch.rand_like(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)

class Bayesian_Dataset(Dataset):
    def __init__(self,file_path,csv_path,opt=1,transform:Optional[Callable]=None,target_transform:Optional[Callable]=None)->None:
        super(Bayesian_Dataset,self).__init__()
        self.imgs_path=file_path
        self.csv_path=csv_path
        data=pd.read_csv(self.csv_path)
        self.test_data=data.iloc[:,0]
        self.test_labels=data.iloc[:,opt]
        self.labels=data.iloc[:,opt]
        self.transform=transform
        self.img_path='img/'
        self.csv_path='target/'
        self.data=data.iloc[:,0]
    
    def __getitem__(self,index:int)->Tuple[Any,Any]:
        imgs_path=self.imgs_path+self.data[index]
        target=int(self.labels[index])
        img=cv2.imread(imgs_path,0)
        img=Image.fromarray(img,mode='L')
        if self.transform is not None:
            img=self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.data)

class GarNet_Dataset(Dataset):
    def __init__(self,file_path,csv_path,opt=1,transform:Optional[Callable]=None,target_transform:Optional[Callable]=None)->None:
        super(GarNet_Dataset,self).__init__()
        self.imgs_path=file_path
        self.csv_path=csv_path
        data=pd.read_csv(self.csv_path)
        self.labels=data.iloc[:,opt]
        self.transform=transform
        self.data=data.iloc[:,0]
        self.video_labels=data.iloc[:,3]
    
    def __getitem__(self,index:int)->Tuple[Any,Any]:
        imgs_path=self.imgs_path+self.data[index]
        target=int(self.labels[index])
        img=cv2.imread(imgs_path,0)
        img=Image.fromarray(img,mode='L')
        if self.transform is not None:
            img=self.transform(img)
        video_label=int(self.video_labels[index])
        return img, target, video_label
    
    def __len__(self):
        return len(self.data)

class OF_Dataset(Dataset):
    def __init__(self,file_path,csv_path,opt=1,transform:Optional[Callable]=None,target_transform:Optional[Callable]=None)->None:
        super(OF_Dataset,self).__init__()
        self.imgs_path=file_path
        self.csv_path=csv_path
        data=pd.read_csv(self.csv_path)
        self.labels=data.iloc[:,opt]
        self.transform=transform
        self.data=data.iloc[:,0]
        self.video_labels=data.iloc[:,3]
    
    def __getitem__(self,index:int)->Tuple[Any,Any]:
        img1_path=self.imgs_path+self.data[index]
        img1=cv2.imread(img1_path, 0)#.transpose((2,0,1))

        img2_path=self.imgs_path+self.data[index+1]
        img2=cv2.imread(img2_path, 0)#.transpose((2,0,1))

        # if self.transform is not None:
        #     img1=self.transform(img1)
        #     img2=self.transform(img2)
        # video_label=int(self.video_labels[index])
        # img1 = img1.permute(2,0,1)
        # img2 = img2.permute(2,0,1)
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        return index, img1, img2
    
    def __len__(self):
        return len(self.data)-1

