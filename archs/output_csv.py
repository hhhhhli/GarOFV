import numpy as np
import csv

def similarity_distance(embeddings,targets,standard_labels,test_labels,standard_std,test_std,number=5):
    f=open('./accuracies.csv','w')
    csv_writer=csv.writer(f)
    csv_writer.writerow(('Name','Accuracy'))
    for t in range(len(test_labels)):
            acc=0
            total=0
            data=test_labels[t]
            distances=np.zeros(len(standard_labels))
            sub_dist=np.zeros(len(standard_labels))
            for n in range(len(standard_labels)):
                dist=np.sum(np.power(standard_labels[n]-data,2))
                distances[n]=dist
            print ('distances:',distances)
            inds=np.where(targets==t+(number+1))[0]
            for i in range (len(inds)):
                    data=embeddings[inds[i]]
                    distances_local=np.zeros(number)
                    for n in range(number):
                        dist_local=np.sum(np.power(standard_labels[n]-data,2))
                        distances_local[n]=dist_local*1+distances[n]*0
                    pred_label=np.argmin(distances_local)
                    if pred_label==t:
                        acc+=1#Location Search
                    total+=1
            accs=(acc/total)*100
            print (f'[{t+1}]Accuracy:{accs}')

def standard_label(embeddings,targets,numbers=10):
    f=open('./standard_labels.csv','w')
    csv_writer=csv.writer(f)
    csv_writer.writerow(('Name','mean_x','mean_y','std_x','std_y'))
    for i in range(numbers):
        inds=np.where(targets==i+1)[0]
        embedding_x=embeddings[inds,0].mean()
        embedding_y=embeddings[inds,1].mean()
        embedding_stdx=embeddings[inds,0].std()
        embedding_stdy=embeddings[inds,1].std()
        csv_writer.writerow((i+1,embedding_x,embedding_y,embedding_stdx,embedding_stdy))
