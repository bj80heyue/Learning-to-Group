#coding=utf-8#
import sys
import Dataset 
import frame
from sys import path
path.append('libsvm/python')
from svmutil import *
import Evaluator
import Evaluate
import Dicision
import random
import numpy as np
import xgboost as xgb
import time

class test:
    def __init__(self):
        self.Album=None
        self.dataset=None
        self.frame=None
        self.num=None
        self.Recall=0
        self.Precision=0
        self.Recall_edge=0
        self.Precision_edge=0 
        self.operatenum=None
        self.dirname=None
        self.history=list()
        self.maxbatch=20
        self.gamma=0.9
        self.K=5           #OP_k步
        self.beta=0.5
        self.randchose=0.5
        self.edge=list()
        self.result=list()
        self.accept=list()
        #0.9 0.6

    def setbatch(self,batchsize):
        self.frame.trainbatch=batchsize

    def loadSimulate(self,dataset):
        f=frame.frame()
        f.loadDataset(dataset)
        self.frame=f
        self.dataset=dataset

    def dtOp(self,history,index):
        return history[index][-1]-history[index+self.K][-1]
    
    def QValue(self,history,index):
        if index == self.maxbatch-self.K-1:
            return  history[index][-2]+self.beta*self.dtOp(history,index)
        else:
            return  history[index][-2]+self.beta*self.dtOp(history,index)+self.gamma*self.QValue(history,index+1)


    def puthistory(self,feature,action,reward_action,Op):
        if len(self.history)>=self.maxbatch:
            #获取Q_value
            value_R=self.QValue(self.history,0)
            #输出该记录
            self.output(self.history[0],value_R)
			#keep the length of history queue constant
            del self.history[0]
            self.history.append((feature,action,reward_action,Op))
        else:
            self.history.append((feature,action,reward_action,Op))
            
    def output(self,batch,value_R):
        if len(batch[0])==3:
            fout=open('data/traindata_Q_p2p','a')
            fout.write(str(value_R))
            fout.write(' '+'0:'+str(batch[1][0]))
            for i in xrange(0,3):
                fout.write(' '+str(i+1)+':'+str(batch[0][i]))
            fout.write('\n')
            fout.close()
        elif len(batch[0])==3+self.frame.k_size:
            fout=open('data/traindata_Q_p2G','a')
            fout.write(str(value_R))
            fout.write(' '+'0:'+str(batch[1][0]))
            for i in xrange(0,3+self.frame.k_size):
                fout.write(' '+str(i+1)+':'+str(batch[0][i]))
            fout.write('\n')
            fout.close()
        else:
            fout=open('data/traindata_Q_G2G','a')
            fout.write(str(value_R))
            fout.write(' '+'0:'+str(batch[1][0]))
            for i in xrange(0,3+2*self.frame.k_size):
                fout.write(' '+str(i+1)+':'+str(batch[0][i]))
            fout.write('\n')
            fout.close()


    def begintest(self,iteration=0):
        model_R_p2p=svm_load_model('model/model_R_p2p.model')
        model_R_p2G=svm_load_model('model/model_R_p2G.model')
        model_R_G2G=svm_load_model('model/model_R_G2G.model')

        model_Q_p2p=xgb.Booster(model_file='model/Q_random_model/model_Q_random_p2p_%d.model'%(iteration))
        model_Q_p2G=xgb.Booster(model_file='model/Q_random_model/model_Q_random_p2G_%d.model'%(iteration))
        model_Q_G2G=xgb.Booster(model_file='model/Q_random_model/model_Q_random_G2G_%d.model'%(iteration))

        data=list(list())
        data.append([0,0,0])
        data_Q=list(list())
        data_Q.append([0,0,0])
        index=0
        reward=0
        decision=Dicision.Dicision()
        t01=time.time()

        while self.frame.checkState():
                package=self.frame.getObservation()
				index += 1
                if type(package)==int:
                        print 'Done!'
                        break
                data[0]=package
                question_type=len(package)
                if question_type==3:        #point-----point
                        tp='P2P'
						#Reward Function
						action_R,confidence,_ = svm_predict([0],data,model_R_p2p,'-b 1')
                        #Reward Value Function: action = 0
                        temp=package[:]
                        temp.insert(0,0)
                        data_Q[0]=temp
                        DM_data=xgb.DMatrix(np.array(data_Q))
                        value_0=model_Q_p2p.predict(DM_data)
                        del temp[0]
                        #Reward Value Function: action = 1
                        temp.insert(0,1)
                        data_Q[0]=temp
                        DM_data=xgb.DMatrix(np.array(data_Q))
                        value_1=model_Q_p2p.predict(DM_data)
						#choose the most awarded action
                        if value_1[0]>=value_0[0]:
                            action=[1]
                        else:
                            action=[0]

                elif question_type==3+self.frame.k_size:     #point-----Group or group---point
                        tp='P2G'
						#Reward Function
						action_R,confidence,_= svm_predict([0],data,model_R_p2G,'-b 1')
                        #Reward Value Function: action = 0
                        temp=package[:]
                        temp.insert(0,0)
                        data_Q[0]=temp
                        DM_data=xgb.DMatrix(np.array(data_Q))
                        value_0=model_Q_p2G.predict(DM_data)
                        del temp[0]
                        #Reward Value Function: action = 1
                        temp.insert(0,1)
                        data_Q[0]=temp
                        DM_data=xgb.DMatrix(np.array(data_Q))
                        value_1=model_Q_p2G.predict(DM_data)
						#choose the most awarded action
                        if value_1[0]>=value_0[0]:
                            action=[1]
                        else:
                            action=[0]
                else:
                        tp='G2G'
						#Reward Function
						action_R,confidence,_ = svm_predict([0],data,model_R_G2G,'-b 1')
                        #Reward Value Function: action = 0
                        temp=package[:]
                        temp.insert(0,0)
                        data_Q[0]=temp
                        DM_data=xgb.DMatrix(np.array(data_Q))
                        value_0=model_Q_G2G.predict(DM_data)
                        del temp[0]
                        #Reward Value Function: action = 1
                        temp.insert(0,1)
                        data_Q[0]=temp
                        DM_data=xgb.DMatrix(np.array(data_Q))
                        value_1=model_Q_G2G.predict(DM_data)
						#choose the most awarded action
                        if value_1[0]>value_0[0]:
                            action=[1]
                        else:
                            action=[0]
                #获取操作量原数量
				# t-lambda processing  (iteration in [1,400])
                if random.random() >= (0.025*iteration):
                    action = [random.randint(0,1)]

				#get reward of the action
                reward_action=10*abs(2*confidence[0][0]-1)

				#get the variance of operate number
                self.frame.Normalize_label()
                operatenum_pre=Evaluator.evaluate(self.dataset.imgID,self.frame.label,[1,2])

				#check the action is True or False
                action_result=self.frame.setPerception(action)
                if action_result==False:
                    reward_action=-reward_action
				#save history
                self.puthistory(package,action,reward_action,operatenum_pre)

		#calculate Metric
        self.frame.Normalize_label()
        self.Recall=Evaluate.Recall(self.dataset.imgID,self.frame.label)
        self.Precision=Evaluate.Precision(self.dataset.imgID,self.frame.label)
        self.operatenum=Evaluator.evaluate(self.dataset.imgID,self.frame.label,[1,2])
        self.Recall_edge=Evaluate.Recall_edge(self.dataset.imgID,self.frame.label,1)
        self.Precision_edge=Evaluate.Precision_edge(self.dataset.imgID,self.frame.label)
        print self.dataset.size,self.Recall_edge,self.Precision_edge,self.operatenum


if __name__=='__main__':
    t=test()
