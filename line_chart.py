import matplotlib.pyplot as plt
import numpy as np
import os


x_axis = [i for i in range(5, 100, 5)]
x_data = [i for i in range(95, 4, -5)]
files = ["Optical-flow_early-stop.txt", "Optical-flow_continuous-perception.txt", "Frame_early-stop.txt", "Frame_continuous-perception.txt"]
result_dir  = "./results"

# fig=plt.figure()
colors = ['#55112F', '#22CED1', '#0033FF', '#7FF554']

for idx, file in enumerate(files):
    f = open(os.path.join(result_dir, file))
    acc_ = [float(line.split(',')[-1][:-1]) for line in f.readlines()]
    acc = [(acc_[i*4] + acc_[i*4+1] + acc_[i*4+2] + acc_[i*4+3]) / 4 for i in range(len(acc_) // 4)]
    # for i in acc:
    #     print(i)
    # print("*******************")
    plt.plot(x_axis, acc, '*--', color = colors[idx], alpha=0.5, linewidth=1, label=file.split('.')[0])
    plt.legend()  #显示上面的label
plt.xticks(x_axis, x_data)


# plt.plot(x_axis_data, y_axis_data, 'b*--', alpha=0.5, linewidth=1, label='acc')#'bo-'表示蓝色实线，数据点实心原点标注
# ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，

# plt.legend()  #显示上面的label
plt.xlabel('bandwidth') #x_label
plt.ylabel('acc')#y_label
 
#plt.ylim(-1,1)#仅设置y轴坐标范围
# plt.show()
plt.savefig('acc.png')
