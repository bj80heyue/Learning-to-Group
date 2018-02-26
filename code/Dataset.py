#coding=utf-8#
import numpy as np
import random
from scipy import spatial as spt
import sys
import math
import os
import shutil
from sys import path
path.append('libsvm/python')
from svmutil import *

class Dataset:
    def __init__(self):
        self.datasetID = 0
        self.imageNameList = list()
        self.rect = list(list())
        self.feature = list(list())
        self.imgID = list()      
        self.Affinity = None
        self.Quality=None           #正脸质量,越低侧脸概率越高
        self.albumnum=None          #类别数量
        self.size = 0
        self.isnoise=None           #判断是否是噪音，as第一层分类器

    def loadfeature(self,featurefileName):
        print 'Load %s'%featurefileName
        fin = open(featurefileName,'r')
        text_data = fin.read().splitlines()
        #训练用格式：
        for i in xrange(0,len(text_data)/2):
            self.imageNameList.append([text_data[i*2]])
            #try
            self.feature.append(map(float,text_data[i*2+1].split()))
            self.imgID.append(0)
            self.size+= 1
        '''
        for i in xrange(0,len(text_data)/3):
            self.imageNameList.append([text_data[i*3]])
            self.rect.append(map(int,text_data[i*3+1].split()))
            self.feature.append(map(float,text_data[i*3+2].split()))
            self.imgID.append(0)
            self.size+= 1
            #print i
        '''
        fin.close()

    def computeAffinity(self):
         print "start computeAffinity..."
         #try
         self.Affinity = 1-spt.distance.pdist(self.feature,'cosine')
         print "finished compute!"
                        
    def computeQuality(self):
        print 'Compute Quality start'
        model=svm_load_model('/media/heyue/8d1c3fac-68d3-4428-af91-bc478fbdd541/ClusterResearch/clusterQNet/model/classify_profile_temp.model')
        #model=svm_load_model('model/model_profile_300.model')
        p_label,p_acc,p_vals=svm_predict([self.imgID[i]==2 for i in range(0,self.size)],self.feature,model,'-b 1')
    #    p_label_b,p_acc_b,p_vals_b=svm_predict([self.imgID[i]==2 for i in range(0,self.size)],self.feature,model_b)
        self.Quality=[x[1] for x in p_vals]
        #self.Quality=[1 for x in xrange(self.size)]
        print 'Compute Quality finished'


class identity_Dataset:
    def __init__(self):
        self.album = list()
        self.albumCount=0
    
    def loadAlbumList(self,albumlistname):
        fin = open(albumlistname,'r')
        text_data = fin.read().splitlines()
        id = 1
        for filepath in text_data:
            temp = Dataset()
            temp.loadfeature(filepath)
            temp.datasetID = id
            id = id+1
            self.albumCount+=1
            self.album.append(temp)
        fin.close()

    #identity_ratio+profile_ratio < 1.0    
    def SimulateDataset(self,albumsize,identity_ratio,profile_ratio):
        dataset = Dataset()
        albumnum=0
        #load identity_image
        identity_size = albumsize*identity_ratio
        identity_num=0
        album_shuffle=range(2,self.albumCount)
        random.shuffle(album_shuffle)
        for identity_index in album_shuffle: #xrange(2,self.albumCount):
            albumnum+=1
            for i in xrange(0,self.album[identity_index].size):
                dataset.imageNameList.append(self.album[identity_index].imageNameList[i])
               # dataset.rect.append(self.album[identity_index].rect[i])
                dataset.feature.append(self.album[identity_index].feature[i])
                dataset.imgID.append(self.album[identity_index].datasetID)
                identity_num+=1
                if identity_num>=identity_size:
                    break
            if identity_num>=identity_size:
                break
            
        #load profile_image
        profile_size = albumsize*profile_ratio
        profile_num=0
        profile_index=1
        album_shuffle=range(0,self.album[profile_index].size)
        random.shuffle(album_shuffle)
        albumnum+=1
        for i in album_shuffle:
            if profile_num>=profile_size:
                break
            dataset.imageNameList.append(self.album[profile_index].imageNameList[i])
           # dataset.rect.append(self.album[profile_index].rect[i])
            dataset.feature.append(self.album[profile_index].feature[i])
            dataset.imgID.append(self.album[profile_index].datasetID)
            profile_num+=1
        
        #load passerby_image
        passerby_size = albumsize-profile_size-identity_size
        passerby_num=0
        passerby_index=0
        album_shuffle=range(0,self.album[passerby_index].size)
        random.shuffle(album_shuffle)
        if passerby_size>0:
            albumnum+=1
        for i in album_shuffle:
            if passerby_num>=passerby_size:
                break
            dataset.imageNameList.append(self.album[passerby_index].imageNameList[i])
           # dataset.rect.append(self.album[passerby_index].rect[i])
            dataset.feature.append(self.album[passerby_index].feature[i])
            dataset.imgID.append(self.album[passerby_index].datasetID)
            passerby_num+=1
        
        dataset.albumnum=albumnum
        dataset.size=albumsize
        return dataset

if __name__ =='__main__':
    b = identity_Dataset()
    b.loadAlbumList('albumList')
    c = b.SimulateDataset(1000,0.5,0.5)   
    c.computeAffinity()
    c.computeQuality()


