#coding=utf-8#
from __future__ import division
import sys

class Dicision:
    def __init__(self):
        self.connection=dict()
        self.x=None
        self.y=None
        #self.threshold=0.1
        self.threshold=0.68  #0.08  pR_44 0.58 0.18
        self.tup=None

    def gettuple(self,Sroot,Droot):
        if self.connection.has_key((Sroot,Droot)):
            return (Sroot,Droot)
        elif self.connection.has_key((Droot,Sroot)):
            return (Droot,Sroot)
        else:
            return 0

    def getAction(self,Sroot,Droot):
        temp=self.gettuple(Sroot,Droot)
        if temp==0:
            self.connection[(Sroot,Droot)]=1
            self.tup=(Sroot,Droot)
        else:
            self.connection[temp]+=1
            self.tup=temp
        self.x=Sroot
        self.y=Droot

    def checkconnection(self,SN,DN):
        smallGroup=min(SN,DN)
       # ratio=self.connection[self.gettuple(self.x,self.y)]/smallGroup
       # if ratio>self.threshold:
        ratio=self.connection[self.gettuple(self.x,self.y)]/(SN*DN)
        if ratio>self.threshold:
            #print 'connnect two group!'
            temp=self.connection[self.tup]
            del self.connection[self.tup]
            for line in self.connection.keys():
                #获取需要改变的类名
                if self.x==line[0]:
                    c=line[1]
                elif self.x==line[1]:
                    c=line[0]
                else:
                    continue
                destline=self.gettuple(c,self.y)
                if destline==0:
                    self.connection[(c,self.y)]=temp
                else:
                    self.connection[destline]+=temp
                del self.connection[line]
            return 1
        else:
           # print 'need more connection'
            return 0

