#coding=utf-8#
import sys
import Dataset
import frame
from sys import path
path.append('libsvm/python')
from svmutil import *
import reward_value_test
import xgboost as xgb
import load_test_data
import traceback
'''
reward Q-value and value-function 
'''

def trainSVMModel():
    print 'Train SVM model start'
    param=svm_parameter('-s 0 -t 0 -b 1')
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
    svm_save_model('model/model_R_p2p_LFW.model',model_p2p)
    svm_save_model('model/model_R_p2G_LFW.model',model_p2G)
    svm_save_model('model/model_R_G2G_LFW.model',model_G2G)
    print 'Train new model finished'

def trainXGBmodel(iteration):
    print 'Train XGBoost model start'
    dtrain_p2p=xgb.DMatrix('data/traindata_Q_p2p')
    dtrain_p2G=xgb.DMatrix('data/traindata_Q_p2G')
    dtrain_G2G=xgb.DMatrix('data/traindata_Q_G2G')
    param={'max_depth':5,'eta':1,'silent':1,'objective':'reg:linear'}
    numround=100
    bst_p2p=xgb.train(param,dtrain_p2p,numround)
    bst_p2G=xgb.train(param,dtrain_p2G,numround)
    bst_G2G=xgb.train(param,dtrain_G2G,numround)
    bst_p2p.save_model('./model/model_Q_inverse_p2p_%d.model'%(iteration))
    bst_p2G.save_model('./model/model_Q_inverse_p2G_%d.model'%(iteration))
    bst_G2G.save_model('./model/model_Q_inverse_G2G_%d.model'%(iteration))
    print 'Train XGBoost model finished'

if __name__=='__main__':
    a=Dataset.identity_Dataset()
    a.loadAlbumList('albumList_train')
    #fin_train=open('data/train_LFW_B','r')
    #trainlist=fin_train.read().splitlines()
    #trainlist=trainlist*10
    data=list(list())
    data.append([0,0,0])
    iteration=0
    batchsize=4
    while iteration<1000:
    #for filepath in trainlist:
        print '====================================================='
        print 'Iter: %d'%iteration
        iteration+=1
        dataset=a.SimulateDataset(1000,0.6,0.4)
        #dataset=load_test_data.load_LFW_dataset(filepath)
        dataset.computeQuality()
        dataset.computeAffinity()
        #do this Dataset
        machine=reward_value_test.test()
        machine.loadSimulate(dataset)
        machine.setbatch(batchsize)
        try:
            machine.begintest(iteration-1)
        except:
            f=open('./log/log.txt','a')
            traceback.print_exc(file=f)
            f.write('='*20)
            f.write('\n')
            f.flush()
            f.close()
            
        batchsize+=1
        #训练新模型
        #trainSVMModel()
        #trainXGBmodel(iteration)
        print machine.operatenum

    print 'Done'

