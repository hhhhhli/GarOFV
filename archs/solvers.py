import torch
import numpy as np
from archs.liteFlowNet.liteFlowNet_run import estimate_batch
from archs.HS import HS


cuda=torch.cuda.is_available()


def train_epoch(train_loader,model,loss_fn,optimizer,cuda,log_interval,metrics,accuracy_metric):
    print('metrics = ', metrics)
    for metric in metrics:
        metric.reset()
    model.train()
    losses=[]
    total_loss=0
    counter=0
    n=0

    for batch_idx,(data,target) in enumerate(train_loader):
        target=target if len(target)>0 else None
        if not type(data) in (tuple,list):
            data=(data,)
        if cuda:
            data=tuple(d.cuda() for d in data)
            if target is not None:
                target=target.cuda()
        
        optimizer.zero_grad()
        outputs=model(*data)

        if not type(outputs) in (tuple,list):
            outputs=(outputs,)
        
        loss_inputs=outputs
        if target is not None:
            target=(target,)
            loss_inputs+=target
        
        loss_outputs=loss_fn(*loss_inputs)
        loss=loss_outputs[0] if type(loss_outputs) in (tuple,list) else loss_outputs
        losses.append(loss.item())
        total_loss+=loss.item()
        loss.backward()
        optimizer.step()
        accuracies=accuracy_metric(*loss_inputs)
        n+=len(accuracies)
        for acc_idx in range (len(accuracies)):
            if accuracies[acc_idx]:
                counter+=1

        for metric in metrics:
            metric(outputs,target,loss_outputs)
        
        if batch_idx%log_interval==0:
            message='Train:[{}/{}({:.0f}%)]\tloss:{:.6f}'.format(batch_idx*len(data[0]),len(train_loader.dataset),100*batch_idx/len(train_loader),np.mean(losses))
            for metric in metrics:
                message+='\t{}:{}'.format(metric.name(),metric.value())
            
            print (message)
            losses=[]
    accuracy=(counter/n)*100
    total_loss/=batch_idx+1
    return total_loss,metrics,accuracy

def test_epoch(val_loader,model,loss_fn,cuda,metrics,accuracy_metric):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
    model.eval()
    val_loss=0
    counter=0
    n=0
    for batch_idx,(data,target) in enumerate(val_loader):
        target=target if len(target)>0 else None
        if not type(data) in (tuple,list):
            data=(data,)
        if cuda:
            data=tuple(d.cuda() for d in data)
            if target is not None:
                target=target.cuda()
        
        outputs=model(*data)

        if not type(outputs) in (tuple,list):
            outputs=(outputs,)
        loss_inputs=outputs
        if target is not None:
            target=(target,)
            loss_inputs+=target
        
        loss_outputs=loss_fn(*loss_inputs)
        loss=loss_outputs[0] if type(loss_outputs) in (tuple,list) else loss_outputs
        val_loss+=loss.item()

        accuracies=accuracy_metric(*loss_inputs)
        n+=len(accuracies)
        for acc_idx in range(len(accuracies)):
            if accuracies[acc_idx]:
                counter+=1
        for metric in metrics:
            metric(outputs,target,loss_outputs)
    accuracy=(counter/n)*100
    return val_loss,metrics,accuracy


def extract_embeddings(dataloader,model):
    with torch.no_grad():
        model.eval()
        embeddings=np.zeros((len(dataloader.dataset),2))
        labels=np.zeros(len(dataloader.dataset))
        k=0
        for images, target in dataloader:
            if cuda:
                images=images.cuda()
            embeddings[k:k+len(images)]=model.get_emdding(images).data.cpu().numpy()
            labels[k:k+len(images)]=target.numpy()
            k+=len(images)
    return embeddings,labels

def extract_embeddings_test(dataloader,model):
    with torch.no_grad():
        model.eval()
        embeddings=np.zeros((len(dataloader.dataset),2))
        labels=np.zeros(len(dataloader.dataset))
        video_labels=np.zeros(len(dataloader.dataset))
        k=0
        for images, target, video_label in dataloader:
            if cuda:
                images=images.cuda()
            embeddings[k:k+len(images)]=model.get_emdding(images).data.cpu().numpy()
            labels[k:k+len(images)]=target.numpy()
            video_labels[k:k+len(images)]=video_label.numpy()
            k+=len(images)
    return embeddings,labels,video_labels

def select_keyFrame(dataloader_OF, video_len, abs_thresh, aver_thresh):
    with torch.no_grad():
        selected_images_idx = np.zeros(len(dataloader_OF.dataset)+1)# set 1 for selected images for 50 videos
        aver_change = None
        k=0
        for iii, img1, img2 in dataloader_OF:
            if k == video_len-1:# 59th img, the end of each video
                k = 0
                aver_change = None
                continue
            if k == 0:
                # selected_images_idx = []
                selected_images_idx[iii] = 1
                idx = k
                aver_change = None
            tenOutput = estimate_batch(img1, img2)
            # dist = torch.sqrt(torch.mul(tenOutput[0], tenOutput[0]).sum() + torch.mul(tenOutput[1], tenOutput[1]).sum())
            dist = torch.sqrt(torch.mul(tenOutput[:,0,:,:], tenOutput[:,0,:,:]).sum() + torch.mul(tenOutput[:,1,:,:], tenOutput[:,1,:,:]).sum())

            # 累积变化达到阈值
            if dist > abs_thresh:
                selected_images_idx[iii+1] = 1
                idx = k
                aver_change = None
            elif aver_change != None:
                # arguments_strOne = img_paths[idx-1]
                # tenOne = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(arguments_strOne))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
                # tenOutput = estimate(tenOne, tenTwo)
                # dist = torch.sqrt(torch.mul(tenOutput[0], tenOutput[0]).sum() + torch.mul(tenOutput[1], tenOutput[1]).sum())
                # 平均变化达到阈值
                if dist > (1 + aver_thresh) * aver_change:# or dist < (1 - aver_thresh) * aver_change:
                    selected_images_idx[iii+1] = 1
                    idx = k
                    aver_change = None
                else:
                    aver_change = (aver_change * (k - idx - 1) + dist) / (k - idx)
            else:
                aver_change = dist

            k += 1 # 1~59
        print('selected')
        

    return selected_images_idx

def Frame_optical(dataloader_OF, video_len):
    with torch.no_grad():
        video_num = (len(dataloader_OF.dataset)+1) // video_len
        frame_optical = np.zeros((video_len-1)*video_num)# 59 optical flow value * 50 videos
        k=0
        video_i = 0
        for iii, img1, img2 in dataloader_OF:
            if k == video_len-1:# 59th img, the end of each video
                k = 0
                video_i += 1
                continue

            tenOutput = estimate_batch(img1, img2)          
            # dist = torch.sqrt(torch.mul(tenOutput[0], tenOutput[0]).sum() + torch.mul(tenOutput[1], tenOutput[1]).sum())
            dist = torch.sqrt(torch.mul(tenOutput[:,0,:,:], tenOutput[:,0,:,:]).sum() + torch.mul(tenOutput[:,1,:,:], tenOutput[:,1,:,:]).sum())
            frame_optical[video_i * (video_len-1) + k] = dist

            k += 1 # 1~59
            # if k == video_len-1:
            #     k = 0

            
        # print('frame_optical = ', frame_optical)
        # print('frame_optical = ', len(frame_optical))
        

    return frame_optical

def Frame_optical_HS(dataloader_OF, video_len):
    video_num = (len(dataloader_OF.dataset)+1) // video_len
    frame_optical = np.zeros((video_len-1)*video_num)# 59 optical flow value * 50 videos
    k=0
    video_i = 0
    for iii, img1, img2 in dataloader_OF:
        if k == video_len-1:# 59th img, the end of each video
            k = 0
            video_i += 1
            continue
        u, v = HS(img1, img2)
        frame_optical[video_i * (video_len-1) + k] = np.sum((u ** 2 + v ** 2) ** 0.5)
        
        k += 1
    
    return frame_optical
