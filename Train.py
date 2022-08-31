#type:ignore

import numpy as np
import torch
from torchvision import transforms
from torch.optim import lr_scheduler

from torch.utils.data import DataLoader
import torch.optim as optim
import time
import os
import argparse
import time
import numpy as np

from archs.datasets import PhySNet_Dataset, TripletMNIST
from archs.models import ResNet18_Embedding, TripletNet
from archs.losses import TripletLoss, TripletAccuracy
from archs.solvers import train_epoch, test_epoch, extract_embeddings
from archs.plots import plot_embeddings


def fit(train_loader,val_loader,model,loss_fn,optimizer,scheduler,n_epochs,cuda,log_interval,train_name,exp_path,time_str,metrics=[],start_epoch=0,par=None):
    accuracy_metric=TripletAccuracy()
    optimizer.step()
    for epoch in range(0,start_epoch):
        scheduler.step()
    for epoch in range(start_epoch,n_epochs):
        scheduler.step()
        train_loss,metrics,accuracy=train_epoch(train_loader,model,loss_fn,optimizer,cuda,log_interval,metrics,accuracy_metric)
        message='Epoch {}/{}. Train set: Average loss:{:.4f} Accuracy:{:.4f}'.format(epoch+1,n_epochs,train_loss,accuracy)
        for metric in metrics:
            message+='\t{}:{}'.format(metric.name(),metric.value())
        
        val_loss,metrics,accuracy=test_epoch(val_loader,model,loss_fn,cuda,metrics,accuracy_metric)
        val_loss/=len(val_loader)
        message+='\nEpoch {}/{}. Validation set: Average loss:{:.4f} Accuracy:{:.4f}'.format(epoch+1,n_epochs,val_loss,accuracy)
        for metric in metrics:
            message+='\t{}:{}'.format(metric.name(),metric.value())
        
        if (epoch+1) % 5 == 0:
            # torch.save(model,par.model_path+train_name+'_epoch%d'+'_%f.pth'%(epoch, time.time()))
            torch.save(model, exp_path+'/'+train_name+'_epoch{}.pth'.format(epoch))
        
        print (message)




##########################################################################
def main():
    
    pars=argparse.ArgumentParser()
    pars.add_argument('--train_mode',type=str,default='train',help='train modes: train, test')
    pars.add_argument('--class_model',type=int,default=1,help='train for shape or weight classification: shape(1), weight(2)')
    pars.add_argument('--rgb_depth',type=str,default='rgb',help='train for shape or weight classification')
    pars.add_argument('--train_file',type=str,default='train_file/',help='train for shape or weight classification')
    pars.add_argument('--vali_file',type=str,default='vali_file/',help='train for shape or weight classification')
    pars.add_argument('--test_file',type=str,default='test_file/',help='train for shape or weight classification')
    pars.add_argument('--img_path',type=str,default='img/',help='train for shape or weight classification')
    pars.add_argument('--csv_path',type=str,default='target/target.csv',help='train for shape or weight classification')
    pars.add_argument('--model_path',type=str,default='Model/',help='train for shape or weight classification')
    pars.add_argument('--exp_name',type=str,default='new_exp/',help='train for shape or weight classification')
    par=pars.parse_args()
    # par = parse(par)

    cuda=torch.cuda.is_available()
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic=True

    # model_path='./Model/'
    rgb_depth_root = './robot_' + par.rgb_depth + '/'
    par.train_file = rgb_depth_root + par.train_file
    par.vali_file = rgb_depth_root + par.vali_file
    par.test_file = rgb_depth_root + par.test_file

    exp_path = rgb_depth_root + par.model_path + par.exp_name + '/'
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    print('1')
    format_str = "%Y%m%d%H%M%S"
    time_str = time.strftime(format_str,time.localtime()) #获取当前格式化过的时间字符串
    # time_array = time.strptime(time_str,format_str)
    # print('e == _epoch+_last_{}.pth'.format(time_str))
    # print('time_str = ', time_str) 
    # exit()

    mean,std=0.00586554,0.03234654
    train_dataset=PhySNet_Dataset(train=True,transform=transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((mean,),(std,))
    ]),opt=par.class_model, par=par)
    test_dataset=PhySNet_Dataset(train=False,transform=transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((mean,),(std,))
    ]),opt=par.class_model, par=par)

    if par.class_model == 1:
        print('Train for shapes classification')
        train_name = 'shape_model'
        physnet_classes=['1','2','3','4','5']
        numbers=[1,2,3,4,5]
    elif par.class_model == 2:
        print('Train for weights classification')
        train_name = 'weight_model'
        physnet_classes=['1','2','3']
        numbers=[1,2,3]

    colors=['#1f77b4','#ff7f01','#2ca02c','#d62728','#9467bd','#7f7f7f','#bcbd22','#17becf','#585957','#7f7f7f']
    fig_path = rgb_depth_root + 'figures/'+ par.exp_name + '/'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    print ('physnet_claasses:',len(physnet_classes))
    print ('colors:',len(colors))

    batch_size=256
    kwargs={'num_workers':1,'pin_memory':True} if cuda else {}
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,**kwargs)
    test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True,**kwargs)


    # if par.train_mode==1:
    triplet_train_dataset=TripletMNIST(train_dataset)
    triplet_test_dataset=TripletMNIST(test_dataset)
    batch_size=28
    kwargs={'num_workers':4,'pin_memory':True} if cuda else {}
    triplet_train_loader=DataLoader(triplet_train_dataset,batch_size=batch_size,shuffle=True,**kwargs)
    triplet_test_loader=DataLoader(triplet_test_dataset,batch_size=batch_size,shuffle=True,**kwargs)
    margin=1
    embedding_net=ResNet18_Embedding()
    print ('embeding_net:',embedding_net)
    model=TripletNet(embedding_net)
    if cuda:
        model=model.cuda()
    lr=1e-3
    params=[]
    print ('---------Params-----------')
    for name,param in model.named_parameters():
        if param.requires_grad==True:
            print ('name:',name)
            params.append(param)
    print ('--------------------------')
    optimizer=optim.Adam(params,lr=lr)
    scheduler=lr_scheduler.StepLR(optimizer,8,gamma=0.1,last_epoch=-1)
    n_epochs=30
    log_interval=100
    loss_fn=TripletLoss(margin)

    if par.train_mode == 'train':
        fit(triplet_train_loader,triplet_test_loader,model,loss_fn,optimizer,scheduler,n_epochs,cuda,log_interval,train_name,exp_path,time_str,par=par)
        # torch.save(model,par.model_path+train_name+'_%f.pth'%time.time())
        torch.save(model, exp_path+train_name+'_last_epoch{}.pth'.format(n_epochs))

        fig_name = fig_path+'last_epoch{}_{}_{}.png'.format(n_epochs, par.train_mode, time_str)
        train_embeddings_triplet,train_labels_triplet=extract_embeddings(train_loader,model)
        plot_embeddings(train_embeddings_triplet,train_labels_triplet,physnet_classes,numbers,colors,fig_name)
        val_embeddings_triplet,val_labels_triplet=extract_embeddings(test_loader,model)
        plot_embeddings(val_embeddings_triplet,val_labels_triplet,physnet_classes,numbers,colors,fig_name)
        
    elif par.train_mode == 'test':
        val_loss,metrics,accuracy=test_epoch(triplet_test_loader,model,loss_fn,cuda,metrics,accuracy_metric)
        val_loss/=len(val_loader)
        message+='\nEpoch {}/{}. Validation set: Average loss:{:.4f} Accuracy:{:.4f}'.format(epoch+1,n_epochs,val_loss,accuracy)
        for metric in metrics:
            message+='\t{}:{}'.format(metric.name(),metric.value())

    print ('--finished!--')

if __name__ == '__main__':
    main()