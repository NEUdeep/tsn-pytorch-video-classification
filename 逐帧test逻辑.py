"""
1，实现高维度数组的连接和拆分。
2，实现video转frame，然后frame在channel维度concat，以及做data augmengtation；
3，为了实现逐帧test，需要从上面的tensor中拿出frame来，所以就有了下面的通道拆分操作；
4，但是tensor不能够迭代，所以需要转numpy；
5，对于高纬度的数组，不能够轻易的按照之前存储为list的时候取frame；
6，按照维度进行通道拆分，拆分按照frame的个数，在channel维度拆分就行；
7，不要考虑用tansform里的转tensor以及其他操作，会傻逼；要懂里面具体的内容；
8，收获很大；
"""


import torch
import numpy as np 
"""
arr1 = np.array([[1,2,3], [4,5,6]])
a = np.split(arr1, [1], axis=0)
print(a)
"""
c = np.arange(24).reshape(1,3,2,4)
i = c.shape[1]
print(c)
b = np.split(c,i,axis=1)
print(b)
print(len(b))
print(b[1])

for j in b:
    frame = torch.from_numpy(j)
    print(frame)


#使用dict的方式，但是优缺点：会合并key的value。

for x in range(1):
    offsets = [int(x)]
print(offsets)

a = [2,2,2]
b = [0.1,0.1,0.3]
d  =dict(zip(b,a))
print(d)

c = (2,0.1)

# 先封装为一个tuple，然后保存为一个list，这样子就可以避免因为dict的key值的相等，而合并不同帧的也测结果。
#最后将这个list保存为txt或者json。
#然后在做帧的预测的时候，直接用一个list保存label；然后使用label和pred组成一个tuple；

z = list()
for i in range(2):

    c = (a[i],b[i])


    z.append(c)
print(z)


import json
 
 
def writeDict(data):
    with open("./data.txt", "w") as f:
    	f.write(json.dumps(data, ensure_ascii=False))
 
if __name__ == '__main__':
 
    dict_1 = {"北京": "BJP", "北京北": "VAP", "北京南": "VNP", "北京东": "BOP", "北京西": "BXP"}
 
    writeDict(dict_1)


d = {}

import pickle
with open("./predict_per_frame-2.file", "rb") as f:
    d = pickle.load(f)

print(d)
for k,v in d.items():
    print(v,k)