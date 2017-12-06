#coding=utf-8
import reward_value_test
import load_test_data
import sys

if __name__=='__main__':
    testdata=list()
    fin_list=open('data/test_trueAlbum','r')
    testdata=fin_list.read().splitlines()

    iteration=400
    for filepath in testdata:
        fout=open('testout/ICCV_supervise_GFW_nonG','a')
		#Initial training environment
        a=reward_value_test.test()
		#Load Dataset
        dataset=load_test_data.load_test_data_set(filepath)
        a.loadSimulate(dataset)
		#start training
        a.begintest(iteration)

        print dataset.size,a.Recall,a.Recall_edge,a.Precision,a.Precision_edge,a.operatenum

