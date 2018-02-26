#coding=utf-8#
import sys
import Dataset
import frame
from sys import path
path.append('libsvm/python')
from svmutil import *

def trainNewModel():
    print 'Train new model start'
    param=svm_parameter('-s 0 -t 0')
    y1,x1=svm_read_problem('data/traindata_p2p')
    y2,x2=svm_read_problem('data/traindata_p2G')
    y3,x3=svm_read_problem('data/traindata_G2G')
    prob1=svm_problem(y1,x1)
    prob2=svm_problem(y2,x2)
    prob3=svm_problem(y3,x3)
    print '....training p2p'
    model_p2p=svm_train(prob1,param)
    print '....training p2G'
    model_p2G=svm_train(prob2,param)
    print '....training G2G'
    model_G2G=svm_train(prob3,param)
    svm_save_model('model/model_p2p.model',model_p2p)
    svm_save_model('model/model_p2G.model',model_p2G)
    svm_save_model('model/model_G2G.model',model_G2G)
    print 'Train new model finished'


if __name__=='__main__':
    a=Dataset.identity_Dataset()
    a.loadAlbumList('albumList_train')
    data=list(list())
    data.append([0,0,0])
    iteration=1
    while iteration<5000:
        print '====================================================='
        print 'Iter: %d'%iteration
        iteration+=1
        dataset=a.SimulateDataset(1000,0.6,0.4)
        dataset.computeQuality()
        dataset.computeAffinity()
        f=frame.frame()
        f.loadDataset(dataset)
        model_p2p=svm_load_model('model/model_p2p.model')
        model_p2G=svm_load_model('model/model_p2G.model')
        model_G2G=svm_load_model('model/model_G2G.model')
        index=1
        while f.checkState():
            package=f.getObservation()
            if type(package)==int:
                print 'Done!'
                break
            data[0]=package
            question_type=len(package)
            if question_type==3:        #point-----point
                action,t1,t2=svm_predict([0],data, model_p2p)
                tp='P2P'
            elif question_type==3+f.k_size:     #point-----Group or group---point
                action,t1,t2=svm_predict([0],data, model_p2G)
                tp='P2G'
            else:
                action,t1,t2=svm_predict([0],data, model_G2G)
                tp='G2G'
            #set action
            if action[0]==1:
                index+=1
            TF=f.setPerception(action)
            if TF==False:
                print action,index,1000,f.albumnum,f.queue.qsize(),tp,f.dataset.imgID[f.S],f.dataset.imgID[f.D],package
        #训练新模型
        #f.Normalize_label()
        #f.showResult()
        trainNewModel()
    print 'Done'

