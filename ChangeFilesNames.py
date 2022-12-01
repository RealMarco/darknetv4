import os
import random

# an engineer can also use simple rename in graphic functions of ubuntu, browse details in https://github.com/RealMarco/InstallSoftwaresonUbuntu/blob/master/useful_commands_of_linux_shell 

path='RPdevkit/RP2022/RealTestset/xml_label' # 


filelist=os.listdir(path)
filelist.sort()
nums_all = [i for i in range(4096)]
nums_list = random.sample(nums_all, 512)

n=0
for i in filelist:
    oldname=path+ '/' + filelist[n]   # os.sep is '/'
    
    # Method 1: re-number the name
    number= int(filelist[n][0:4])-4736
    # Method 2: random select the name
    # number = nums_list[n]
    
    newname=path + '/' + str(number).zfill(4)+ filelist[n][4:]
    os.rename(oldname,newname)   #用os模块中的rename方法对文件改名
    #print(oldname,'======>',newname)
    
    n+=1
