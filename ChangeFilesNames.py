import os

# an engineer can also use simple rename in graphic functions of ubuntu, browse details in https://github.com/RealMarco/InstallSoftwaresonUbuntu/blob/master/useful_commands_of_linux_shell 

path='RPdevkit/RP2022/DepthImages'     

filelist=os.listdir(path)

n=0
for i in filelist:
    oldname=path+ '/' + filelist[n]   # os.sep is '/'
    number= int(filelist[n][0:4])+96
    newname=path + '/' + str(number).zfill(4)+ filelist[n][4:]
    os.rename(oldname,newname)   #用os模块中的rename方法对文件改名
    #print(oldname,'======>',newname)
    
    n+=1
