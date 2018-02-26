#coding=utf-8#
from __future__ import division 

#默认1:路人　2:侧脸及误检　3+:grouped
#召回率
def Recall(gt,result):
    sum_C=0
    sum_n=0
    sum_noise=0
    F=len(gt)
    for i in xrange(0,len(result)):
        if result[i]!=1 and result[i]!=2:
            sum_C+=1
            if gt[i]==1 or gt[i]==2:
                sum_n+=1
        if gt[i]==1 or gt[i]==2:
            sum_noise+=1
    print sum_C,sum_n,sum_noise,F
    print 'Recall:%d\tSum:%d'%(sum_C,F)
    R=(sum_C-sum_n)/(F-sum_noise)
    return R

def Precision(gt,result):
    dict_group=dict()
    sum_C=0
    for i in xrange(0,len(result)):
        groupID=result[i]
        if dict_group.has_key(groupID):
            dict_group[groupID].append(gt[i])
        else:
            dict_group[groupID]=list()
            dict_group[groupID].append(gt[i])
    sum_misgroup=0
    #对每个group分别求众数
    if dict_group.has_key(1) and dict_group.has_key(2):
        pd=1
        print '1:%d\t2:%d'%(len(dict_group[1]),len(dict_group[2]))
    elif dict_group.has_key(1)==False and dict_group.has_key(2)==True:
        pd=2
    elif dict_group.has_key(1)==True and dict_group.has_key(2)==False:
        pd=2
    else:
        pd=3

    for i in xrange(3,len(dict_group)+pd):
        arr_appear = dict((a, dict_group[i].count(a)) for a in dict_group[i]);  # 统计各个元素出现的次数  
        mode=max(arr_appear.values())
        sum_misgroup+=(len(dict_group[i])-mode)
        sum_C+=len(dict_group[i])
        print i,(len(dict_group[i])-mode),len(dict_group[i]),mode/len(dict_group[i])
    print 'Precision:', sum_misgroup,sum_C 
    if sum_C==0:
        return 0.0
    P=1-sum_misgroup/sum_C
    return P

def misedge(arr):
    if len(arr)==1:
        return 0
    else:
        sum_edge=0
        for i in xrange(0,len(arr)):
            for j in xrange(i+1,len(arr)):
                sum_edge+=arr[i]*arr[j]
        return sum_edge

def Recall_edge(gt,result,ignore):
    dict_label=dict()
    for i in xrange(0,len(gt)):
        if dict_label.has_key(gt[i]):
            if dict_label[gt[i]].has_key(result[i]):
                dict_label[gt[i]][result[i]].append(i)
            else:
                dict_label[gt[i]][result[i]]=list()
                dict_label[gt[i]][result[i]].append(i)
        else:
            dict_label[gt[i]]=dict()
            dict_label[gt[i]][result[i]]=list()
            dict_label[gt[i]][result[i]].append(i)
    mis_edge=0
    sum_edge=0
    for t in dict_label:
        if ignore==1 and (t==1 or t==2):
            continue
        else:
            arr=list()
            for ti in dict_label[t]:
                arr.append(len(dict_label[t][ti]))
                if ti==1 or ti==2:
                    ni=len(dict_label[t][ti])
                    mis_edge+=ni*(ni-1)/2
            mis_edge+=misedge(arr)
            n=sum(arr)
            sum_edge+=n*(n-1)/2
    return 1-mis_edge/sum_edge

def Precision_edge(gt,result):
    dict_label=dict()
    for i in xrange(0,len(result)):
        if dict_label.has_key(result[i]):
            if dict_label[result[i]].has_key(gt[i]):
                dict_label[result[i]][gt[i]].append(i)
            else:
                dict_label[result[i]][gt[i]]=list()
                dict_label[result[i]][gt[i]].append(i)
        else:
            dict_label[result[i]]=dict()
            dict_label[result[i]][gt[i]]=list()
            dict_label[result[i]][gt[i]].append(i)
    precision_list=list()
    num_label=list()
    sum_true=0
    sum_all=0
    for t in dict_label:
        if t==1 or t==2:
            continue
        else:
            arr=list()
            for ti in dict_label[t]:
                arr.append(len(dict_label[t][ti]))
            dm=max(arr)
            dn=sum(arr)
            edge_true=0
            for k in arr: 
                edge_true+=k*(k-1)
            edge_all=dn*(dn-1)

            sum_true+=edge_true
            sum_all+=edge_all
            precision_list.append(edge_true/edge_all)
   # print precision_list
    print 'Precision_edge',sum_true,sum_all
   #return sum(precision_list)/len(precision_list)
    if sum_all==0:
        return 0
    return sum_true/sum_all

if __name__=='__main__':
    #r=[3,3,4,4,4,4,5,5,4,4,4,4,3]
    #gt=[3,3,3,3,3,3,4,4,5,5,5,5,5]
    fin_gt=open('./data/OPPO/gt.out','r')
    gt=map(int,fin_gt.read().splitlines())
    fin_result=open('./result.txt','r')
    result=map(int,fin_result.read().splitlines())
    
    recall=Recall(gt,result)
    precision=Precision(gt,result)
    recall_edge=Recall_edge(gt,result,1)
    precision_edge=Precision_edge(gt,result)
    print 'Recall_edge:%f\nPrecision_edge:%f\n'%(recall_edge,precision_edge)  
