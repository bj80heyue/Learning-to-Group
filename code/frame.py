#coding=utf-8#
from __future__ import division
import sys
import Queue
from  Dataset import *
import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.path import Path
import matplotlib.patches as patches
from pylab import *
import time
import random


from libshow import libshow
import shutil

#Index from 0 to n*(n-1)/2-1
def findXY(Index,n):
    k=1
    while Index>=(n-k):
        Index-=(n-k)
        k+=1
    return k-1,k+Index

class frame:
    def __init__(self):
        self.size=0
        self.queue=None             #边表队列n*n个元素
        self.dataset=None           #模拟相册
        self.UnionFind=list()       #并查集
        self.groupsize=list()
        self.gt=None
        self.label=list()           #结果输出list
        self.AffinityMatrix=None    #相关矩阵
        self.Quality=None           #人脸质量
        self.k_size=5               #超参数，附加个数
        self.trainbatch=15         #每收集batch重新训练
        self.traingetnum=0          #一轮训练中已经获得的训练数据条目
        self.S=None                 #上一步中取得的节点S
        self.D=None                 #上一步中取得的节点D
        self.package=None           #记录上一步取出的特征
        self.albumnum=0             #该相册中照片分类数量
        self.Threshold=0.60  #相关度阈值0.60   vis 72
        self.Threshold_Quality=0.50 #质量分离阈值
        self.Threshold_Passerby=3   #路人阈值
        self.getlabel=None          #从外部接受label进行处理

    def loadDataset(self,dataset):
        print "load dataset start"
        self.dataset=dataset
        #dataset.computeAffinity()
        #dataset.computeQuality()
        self.size=dataset.size
        self.gt=dataset.imgID
        self.albumnum=dataset.albumnum
        self.AffinityMatrix=spt.distance.squareform(dataset.Affinity)
        self.Quality=dataset.Quality
        Affinity=dataset.Affinity
        self.UnionFind=range(0,self.size)
        self.groupsize=[1]*self.size
        self.label=list([0]*self.size)
        #获得降序排列的边
        print "....translate Queue start"
        self.queue=Queue.Queue(self.size*self.size)
        DescentIndex=sorted(range(len(Affinity)),key=lambda k: Affinity[k],reverse=True)
        for index in DescentIndex:
            if Affinity[index]>self.Threshold:
                self.queue.put(findXY(index,self.size))
            else:
                break
        #random candidate list
        '''
        temp=[]
        for index in DescentIndex:
            if Affinity[index]>self.Threshold:
                temp.append(findXY(index,self.size))
            else:
                break
        random.shuffle(temp)
        for item in temp:
            self.queue.put(item)
        '''
        print "....translate Queue finished"
        print "load dataset finished"

    def getRect(self,index,Rsize):
        rect_source=self.dataset.rect[index]
        #d=45
        rect_end=[0,0,0,0]
        ratio=0.7
        dw=int((rect_source[2]-rect_source[0])*ratio)
        dh=int((rect_source[3]-rect_source[1])*ratio)

        rect_end[0]=min(max(rect_source[0]-dw,0),Rsize[0]-1)
        rect_end[1]=min(max(rect_source[1]-dh,0),Rsize[1]-1)
        rect_end[2]=min(max(rect_source[2]+dw,0),Rsize[0]-1)
        rect_end[3]=min(max(rect_source[3]+dh,0),Rsize[1]-1)
        return rect_end

    def showQueue(self,n):
        print "showQueue start"
        shutil.rmtree('QueueVisualize')
        os.mkdir('QueueVisualize')
        for index in xrange(1,n+1):
            imageLink=self.queue.get()
            fig=plt.figure()
            path_a=self.dataset.imageNameList[imageLink[0]]
            path_b=self.dataset.imageNameList[imageLink[1]]
            im_a=Image.open(path_a[0])
            im_b=Image.open(path_b[0])
            im_a=im_a.crop(self.getRect(imageLink[0],im_a.size))
            im_b=im_b.crop(self.getRect(imageLink[1],im_b.size))
            im_a=im_a.resize((256,256))
            im_b=im_b.resize((256,256))
            a=fig.add_subplot(2,1,0)
            imgplot = plt.imshow(im_a)
            imgplot.axes.get_xaxis().set_visible(False)
            imgplot.axes.get_yaxis().set_visible(False)
            b=fig.add_subplot(2,1,1)
            imgplot = plt.imshow(im_b)
            imgplot.axes.get_xaxis().set_visible(False)
            imgplot.axes.get_yaxis().set_visible(False)
            plt.savefig('./QueueVisualize/'+str(index)+'.jpg')
            plt.close()
        print "showQueue finished"
    
    def showResult(self):
        print "showResult start"
        shutil.rmtree(r'Visualize')
        os.mkdir('Visualize')
        path_head='/home/heyue/Documents/DATA/miningimage2'
        for index in xrange(0,self.size):
            path=self.dataset.imageNameList[index]
            path_back=path[0]
            image=Image.open(path_head+path_back[1:])
       #     image=Image.open(path_back)
            #move　不用下一行
        #    image=image.crop(self.getRect(index,image.size))
            image=image.resize((256,256))
            label=self.label[index]
            if os.path.exists('./Visualize/'+str(label)):
                pass
            else:
                os.mkdir('./Visualize/'+str(label))
            image.save('./Visualize/'+str(label)+'/'+str(index)+'.jpg','JPEG')
            #print index,self.size
    # thumbnail=libshow()
     #   thumbnail.run()
        print "showResult Done"

    def median(self,numberlist):
        numberlist=sorted(numberlist)
        length=len(numberlist)
        if length%2==1:
            return numberlist[int(length/2)]
        else:
            return (numberlist[int(length/2)-1]+numberlist[int(length/2)])/2.0

    def midAffinity(self,group_element):
        score=list()
        for i in xrange(0,len(group_element)):
            affinitylist=list()
            for j in xrange(0,len(group_element)):
                if j!=i:
                    affinitylist.append(self.AffinityMatrix[group_element[i]][group_element[j]])
            score.append(self.median(affinitylist))
        return score

    def Recommend(self,candidate_list,aim_list,num):
        score=list()
        for x in candidate_list:
            affinitylist=list()
            for y in aim_list:
                affinitylist.append(self.AffinityMatrix[x][y])
            score.append(self.median(affinitylist))
        DescentIndex=sorted(range(len(score)),key=lambda k: score[k],reverse=True)
        recommend_index=[candidate_list[i] for i in DescentIndex[:num]]
        recommend_score=[score[i] for i in DescentIndex[:num]]
        return recommend_index,recommend_score

    def showResult_Order(self,dirname):
        print 'show result in order'
        os.mkdir(dirname)
        dict_label=dict()
        dict_element_order=dict()
        for i in xrange(0,self.size):
            root_i=self.label[i]
            if dict_label.has_key(root_i):
                dict_label[root_i].append(i)
            else:
                dict_label[root_i]=list()
                dict_element_order[root_i]=list()
                dict_label[root_i].append(i)
        for root in dict_label:
            if root==1 or root==2:
                continue
            print 'group',root
            #os.mkdir(dirname+'/'+str(root))
            #fout_gt=open(dirname+'/'+str(root)+'/gt.txt','w')
            score=self.midAffinity(dict_label[root])     #返回每个图片在此group中的得分
            DescentIndex=sorted(range(len(score)),key=lambda k: score[k],reverse=True)

            side=math.ceil(math.sqrt(len(score)))
            fig=plt.figure(figsize=(20.0,15.0))
            for i in xrange(0,len(DescentIndex)):
                #最后一行留给推荐系统
                a=fig.add_subplot(side+2,side,i+1)
                index=dict_label[root][DescentIndex[i]]
                #显示图片编号2016.10.24
                #a.set_title(str(self.Quality[index]))
                path=self.dataset.imageNameList[index]
                path_back=path[0]
                path_next=path_back.replace('/home/sensetime/renren/pein0119_test/','/home/heyue/Downloads/clusterQNet/data/renren_new/')
                path_next=path_next.replace('/data3/ckd/RenrenFeature/','/home/heyue/Downloads/clusterQNet/data/renren_new/')
                image=Image.open(path_next)
                #image=Image.open(path[0])
                mode_pre=image.mode
                if image.mode!='RGB':
                    image=image.convert('RGB')
                #print mode_pre+'-->'+image.mode
                #move　不用下一行
                image=image.crop(self.getRect(index,image.size))
                image=image.resize((256,256))
                #存下小图
                #image.save(dirname+'/'+str(root)+'/'+str(i)+'.jpg')
                #fout_gt.write('%d %d\n'%(i,self.gt[index]))

                implot=plt.imshow(image)
                #查询gt
                gt=self.gt[index]
                first=self.gt[dict_label[root][DescentIndex[0]]]               
                flag=0
                if first==1 or first==2:
                    flag=1
                if gt!=first and flag==0:
                    patch=Rectangle((0,0),255,255,facecolor='None',edgecolor='yellow',linewidth=10)
                    plt.gca().add_patch(patch)
                implot.axes.get_xaxis().set_visible(False)
                implot.axes.get_yaxis().set_visible(False)

            #显示推荐图片
            if root!=1 and root!=2:
                if dict_label.has_key(1) and dict_label.has_key(2):
                    candidate_list=dict_label[1]+dict_label[2]
                elif dict_label.has_key(1):
                    candidate_list=dict_label[1]
                else:
                    candidate_list=dict_label[2]

                candidate,score=self.Recommend(candidate_list,dict_label[root],int(side*2))
                for i in xrange(0,len(candidate)):
                    a=fig.add_subplot(side+2,side,side*side+i+1)
                    index=candidate[i]
                    #显示图片编号2016.10.24
                    #a.set_title(str(index))
                    path=self.dataset.imageNameList[index]
                    path_back=path[0]
                    path_next=path_back.replace('/home/sensetime/renren/pein0119_test/','/home/heyue/Downloads/clusterQNet/data/renren_new/')
                    path_next=path_next.replace('/data3/ckd/RenrenFeature/','/home/heyue/Downloads/clusterQNet/data/renren_new/')
                    image=Image.open(path_next)
                    #move　不用下一行
                    image=image.crop(self.getRect(index,image.size))
                    image=image.resize((256,256))
                    implot=plt.imshow(image)
                    #查询gt
                    gt=self.gt[index]
                    if gt==self.gt[dict_label[root][DescentIndex[0]]] and gt!=1 and gt!=2:
                        patch=Rectangle((0,0),255,255,facecolor='None',edgecolor='red',linewidth=10)
                        plt.gca().add_patch(patch)
                    implot.axes.get_xaxis().set_visible(False)
                    implot.axes.get_yaxis().set_visible(False)
                    if i<side:
                        title_str='--------------------------------------------------------------------------------------------------'
                        a.set_title(r'----------------------------------------------------------------------------------------------',fontsize=20,color='r')
                        #a.set_title(title_str+str(self.Quality[index])+title_str,fontsize=20,color='r')
            plt.savefig(dirname+'/'+str(root)+'.jpg')
            plt.close(fig)
        print 'show result finished'
                

    def findroot(self,index):
        root=index
        #查询root
        while(root!=self.UnionFind[root]):
            root=self.UnionFind[root]
        #路径压缩
        i=index
        while(i!=root):
            temp=self.UnionFind[i]
            self.UnionFind[i]=root
            i=temp
        return root

    def join(self,x,y):
        rx=self.findroot(x)
        ry=self.findroot(y)
        if(rx!=ry):
            self.UnionFind[rx]=ry
            self.groupsize[ry]+=self.groupsize[rx]
    
    #查询x当前Group中的成员
    def findGroupMember(self,x):
        rx=self.findroot(x)
        GML=list()
        for i in xrange(0,self.size):
            if self.findroot(i)==rx:
                GML.append(i)
        return GML
    
    #判断是否在Group中
    def InGroup(self,x):
        if x!=self.UnionFind[x]:
            return True
        else:
            if len(self.findGroupMember(x))==1:
                return False
            else:
                return True
    def get_Knearest(self,x):
        GML=self.findGroupMember(x)
        DM_Affinity=[self.AffinityMatrix[x][i] for i in GML]
        DescentIndex=sorted(range(len(DM_Affinity)),key=lambda k: DM_Affinity[k],reverse=True)
        while len(DescentIndex)<self.k_size:
            DescentIndex.append(DescentIndex[-1])
        KDescentIndex=DescentIndex[0:self.k_size]
        K_select=[GML[i] for i in KDescentIndex]
        return K_select      

    def outputdata(self):
	feature = list()
	S = self.S
        type_s=self.InGroup(S)
	if type_s == False:
	    for i in xrange(5):
		if len(self.dataset.feature[S])>256:
	            feature.append(self.dataset.feature[S][3:])
		else:
	            feature.append(self.dataset.feature[S])
	elif type_s == True:
	    K_select = self.get_Knearest(S)
            for index in K_select:
		if len(self.dataset.feature[index])>256:
	            feature.append(self.dataset.feature[index][3:])
		else:
	            feature.append(self.dataset.feature[index])
	D = self.D
        type_d=self.InGroup(D)
	if type_d == False:
            for i in xrange(5):
		if len(self.dataset.feature[D])>256:
	            feature.append(self.dataset.feature[D][3:])
		else:
	            feature.append(self.dataset.feature[D])
	elif type_d == True:
	    K_select = self.get_Knearest(D)
	    for index in K_select:
		if len(self.dataset.feature[index])>256:
	            feature.append(self.dataset.feature[index][3:])
		else:
	            feature.append(self.dataset.feature[index])

	'''
	if self.gt[S]!=self.gt[D] and (S+D)>4:
            flag = 0
	elif self.gt[S]==self.gt[D]:
            flag = 1
	fout = open('./MLPing/MLPdata.txt','a')
	fout.write('%d\n'%(flag))
	for ft in feature:
            ft = ft[3:]
            fout.write(' '.join(map(str,ft))+'\n')
	fout.close()
	'''
	return feature
		
    def getObservation(self):
        if self.queue.empty()==False:
            linkage=self.queue.get()
        else:
            return 0
        while self.findroot(linkage[0])==self.findroot(linkage[1]):
            if self.queue.empty():
                return 0
            linkage=self.queue.get()
            
            #print 'delete edge'
        S=linkage[0]
        D=linkage[1]
        if self.AffinityMatrix[S][D]<self.Threshold:
            self.queue=Queue.Queue()
            return 0
        self.S=S
        self.D=D
        #判断S,D类型：Point / Group
        type_s=self.InGroup(S)
        type_d=self.InGroup(D)
        #针对S\D类型输出参数
        package=[self.Quality[S],self.Quality[D],self.AffinityMatrix[S][D]]
        if type_s==False and type_d==False:     #point----point
            self.package=package
            return package
        elif type_s==False and type_d==True: #point----group
            K_select=self.get_Knearest(D)
            for index in K_select:
                package.append(self.AffinityMatrix[S][index])
            self.package=package
            return package
        elif type_s==True and type_d==False: #group----point
            K_select=self.get_Knearest(S)
            package=[self.Quality[D],self.Quality[S],self.AffinityMatrix[D][S]]
            for index in K_select:
                package.append(self.AffinityMatrix[D][index])
            self.package=package
            temp=self.D
            self.D=self.S
            self.S=temp
            return package
        else:                                   #group----group
            K_select_S=self.get_Knearest(S)
            K_select_D=self.get_Knearest(D)
            for index in K_select_D:
                package.append(self.AffinityMatrix[S][index])
            for index in K_select_S:
                package.append(self.AffinityMatrix[D][index])
            self.package=package
            return package

    def save_traindata(self,gt):
        #print 'save train_data begin'
        data_type=len(self.package)
        if data_type==3:                    #3
            fout=open('data/traindata_p2p','a')
            fout.write(str(gt))
            for i in xrange(0,len(self.package)):
                fout.write(" "+str(i+1)+":"+str(self.package[i]))
            fout.write("\n")
            fout.close()
        elif data_type==3+self.k_size:
            fout=open('data/traindata_p2G','a')
            fout.write(str(gt))
            for i in xrange(0,len(self.package)):
                fout.write(" "+str(i+1)+":"+str(self.package[i]))
            fout.write("\n")
            fout.close()
        else:
            fout=open('data/traindata_G2G','a')
            fout.write(str(gt))
            for i in xrange(0,len(self.package)):
                fout.write(" "+str(i+1)+":"+str(self.package[i]))
            fout.write("\n")
            fout.close()
        #print 'save train_data finished'


    #action=１or 0　１加入　０　拒绝
    def setPerception(self,action):
        if action[0]==1:        #加入
            self.join(self.S,self.D)
            if self.dataset.imgID[self.S]==self.dataset.imgID[self.D]:
                return True
            elif self.dataset.imgID[self.S]==1 and self.dataset.imgID[self.D]==2:
                return True
            elif self.dataset.imgID[self.S]==2 and self.dataset.imgID[self.D]==1:
                return True
            else:
                pp=0
                #self.save_traindata(pp)
                self.traingetnum+=1
                return False
        else:                   #拒绝
            if self.dataset.imgID[self.S]==1 and self.dataset.imgID[self.D]==1:
                return True
            elif self.dataset.imgID[self.S]==2 and self.dataset.imgID[self.D]==2:
                return True
            elif self.dataset.imgID[self.S]==self.dataset.imgID[self.D]:
                #self.save_traindata(1)
                self.traingetnum+=0.2
                return False
            else:
                return True
                pass
    def checkState(self):
        if self.queue.empty() or self.trainbatch<=self.traingetnum:
            return False
        else: 
            return True

    #正规化label:1 路人　２侧脸　3++　类别
    def Normalize_label(self):
        dict_group=dict()
        dict_group_Q=dict()
        for i in xrange(0,self.size):
            if self.getlabel!=None:
                root_i=self.getlabel[i]
            else:
                root_i=self.findroot(i)

            if dict_group.has_key(root_i):
                dict_group[root_i].append(i)
                dict_group_Q[root_i].append(self.Quality[i])
            else:
                dict_group[root_i]=list()
                dict_group_Q[root_i]=list()
                dict_group[root_i].append(i)
                dict_group_Q[root_i].append(self.Quality[i])
        #calculate Average Quality
        dict_average_Q=dict()
        dict_index=dict()
        indexbase=3
        passerby_num=0
        profile_num=0
        for root_i in dict_group_Q:
            dict_average_Q[root_i]=sum(dict_group_Q[root_i])/len(dict_group_Q[root_i])
            #设定侧脸阈值
            if dict_average_Q[root_i]<self.Threshold_Quality:#0.500
                #print 'Profile_filter:',dict_average_Q[root_i],len(dict_group[root_i])
                for i in dict_group[root_i]:
                   # if self.label[i]==0:
                    self.label[i]=2
                    profile_num+=1
            #设定路人阈值
            elif len(dict_group[root_i])<self.Threshold_Passerby:
                passerby_num+=1
                for i in dict_group[root_i]:
                    if self.label[i]==0:
                        self.label[i]=1
            else:
                if dict_index.has_key(root_i):
                    pass
                else:
                    dict_index[root_i]=indexbase
                    indexbase+=1
                    print indexbase,dict_average_Q[root_i]
                for i in dict_group[root_i]:
                    #if self.label[i]==0:
                    self.label[i]=dict_index[root_i]
                #print dict_index[root_i],dict_average_Q[root_i],len(dict_group[root_i])
        print 'profile num:'+str(profile_num)
        print 'passerby num:'+str(passerby_num)


    #生成预训练数据
    def preTrainData(self,n):
        for i in xrange(0,n):
            d=self.getObservation()
            if d==0:
                break
            if self.dataset.imgID[self.S]==self.dataset.imgID[self.D]:
                gt=1
                self.join(self.S,self.D)
            else:
                gt=0
            self.save_traindata(gt)

if __name__=='__main__':
    b=identity_Dataset()
    b.loadAlbumList(r'albumList_train')
    c = b.SimulateDataset(1000,0.5,0.5)
    c.computeAffinity()
    c.computeQuality()
    f=frame()
    f.loadDataset(c)
    f.preTrainData(900)
