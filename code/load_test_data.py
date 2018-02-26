from Dataset import Dataset

def load_test_data_set(filename):
    fin = open(filename, 'r')
    lines = fin.read().splitlines()
    #data_num = len(lines) / 3
    data_num = len(lines) / 4
    dataset = Dataset()
    dataset.Quality=[]
    for i in xrange(data_num):
        '''
        dataset.imageNameList.append([lines[i * 3]])
        dataset.imgID.append(int(lines[i * 3 + 1]))
        dataset.feature.append(map(float, lines[i * 3 + 2].split()))
        '''
        dataset.imageNameList.append([lines[i * 4]])
        dataset.rect.append(map(int,lines[i*4+1].split()))
        dataset.imgID.append(int(lines[i * 4 + 2]))
        dataset.feature.append(map(float, lines[i * 4 + 3].split()))
        #dataset.feature[-1]=dataset.feature[-1][:128]
        #dataset.Quality.append(1.0)
        dataset.size += 1
    dataset.computeAffinity()
    dataset.computeQuality()
    '''
    #finetune Quality
    quality_path='data/Quality/'+filename.split('/')[-1]
    fin_q=open(quality_path,'r')
    data_q=fin_q.read().splitlines()
    fin_q.close()
    dataset.Quality=map(float,data_q)
    if len(dataset.Quality)!=dataset.size:
        return None
    '''
    return dataset

def load_HP_dataset(noise=False):
    fin=open('/media/heyue/8d1c3fac-68d3-4428-af91-bc478fbdd541/ClusterResearch/clusterQNet/data/HP_model_5_feature.txt','r')
    lines=fin.read().splitlines()
    data_num=len(lines)/5
    dataset=Dataset()
    
    for i in xrange(data_num):
        dataset.imageNameList.append(['data/HP/'+lines[i*5]])
        dataset.rect.append([0,0,100,100])
        dataset.imgID.append(int(lines[i*5+4]))
        dataset.feature.append(map(float,lines[i*5+2].split()))
        dataset.size+=1
        if dataset.imgID[-1]==1 and noise==False:
            dataset.imageNameList.pop()
            dataset.rect.pop()
            dataset.imgID.pop()
            dataset.feature.pop()
            dataset.size-=1
    dataset.computeAffinity()
    #dataset.computeQuality()
    dataset.Quality=[1.0 for i in xrange(dataset.size)]
    return dataset

def load_lfw_dataset():
    fin=open('data/Foreign_Dataset/LFW_model_5_feature.txt','r')
    lines=fin.read().splitlines()
    data_num=len(lines)/2
    dataset=Dataset()
    index=3
    gt=dict()
    num=0
    gt_dict=dict()

    for i in xrange(data_num):
        dataset.imageNameList.append(['data/Foreign_Dataset/LFW_align/'+lines[i*2]])
        dataset.rect.append([0,0,178,218])
        name=lines[i*2].split('/')[0]

        if gt_dict.has_key(name):
            gt_dict[name].append(i)
        else:
            gt_dict[name]=list()

        if gt.has_key(name):
            dataset.imgID.append(gt[name])
        else:
            gt[name]=index
            index+=1
            dataset.imgID.append(gt[name])
        dataset.feature.append(map(float,lines[i*2+1].split()))
        dataset.size+=1
    #dataset.computeAffinity()
    dataset.Quality=[0.99 for i in xrange(dataset.size)]
    return dataset,gt_dict

def load_LFW_dataset(filepath):
    fin=open(filepath,'r')
    lines=fin.read().splitlines()
    data_num=len(lines)/3
    dataset=Dataset()

    for i in xrange(data_num):
        dataset.imageNameList.append([lines[i*3]])
        dataset.imgID.append(int(lines[i*3+1]))
        dataset.rect.append([0,0,178,218])
        dataset.feature.append(map(float,lines[i*3+2].split()))
        dataset.size+=1
    dataset.computeAffinity()
    #dataset.computeQuality()
    dataset.Quality=[0.9 for i in xrange(dataset.size)]
    return dataset

def load_cpf_dataset():
    fin=open('data/Foreign_Dataset/cfp_model_5_feature.txt','r')
    lines=fin.read().splitlines()
    data_num=len(lines)/2
    dataset=Dataset()
    index=3
    gt=dict()
    num=0
    gt_dict=dict()

    for i in xrange(data_num):
        dataset.imageNameList.append(['data/Foreign_Dataset/cfp_align/'+lines[i*2]])
        name=lines[i*2].split('/')[0]
        tp=lines[i*2].split('/')[1]
        if gt_dict.has_key(name):
            gt_dict[name][tp].append(i)
        else:
            gt_dict[name]=dict()
            gt_dict[name]['frontal']=list()
            gt_dict[name]['profile']=list()
            gt_dict[name][tp].append(i)

        dataset.rect.append([0,0,178,218])
        if gt.has_key(name):
            dataset.imgID.append(gt[name])
        else:
            gt[name]=index
            index+=1
            dataset.imgID.append(gt[name])
        dataset.feature.append(map(float,lines[i*2+1].split()))
        dataset.size+=1
    dataset.computeAffinity()
    dataset.computeQuality()
    #dataset.Quality=[0.99 for i in xrange(dataset.size)]
    return dataset,gt_dict

def load_MV_dataset(num):
    fin=open('data/Foreign_Movie_model_5_feature.txt','r')
    lines=fin.read().splitlines()
    data_num=len(lines)/2
    dataset=Dataset()
    data_num=num
    for i in xrange(data_num):
        dataset.imageNameList.append(['data/Foreign_Movie_Face/'+lines[2*i]])
        dataset.rect.append([0,0,178,218])
        dataset.imgID.append(1)
        dataset.feature.append(map(float,lines[2*i+1].split()))
        dataset.size+=1
    dataset.computeAffinity()
    dataset.computeQuality()
    return dataset

def load_nongt_nonquality(filename):
    fin = open(filename, 'r')
    lines = fin.read().splitlines()
    data_num = len(lines) / 3
    dataset = Dataset()

    for i in xrange(data_num):
        dataset.imageNameList.append([lines[i * 3]])
        dataset.rect.append(map(int,lines[i*3+1].split()))
        dataset.feature.append(map(float, lines[i * 3 + 2].split()))
        dataset.imgID.append(0)
        dataset.size += 1
    dataset.computeAffinity()
    dataset.Quality=[0.99 for i in xrange(dataset.size)]
    #dataset.computeQuality()
    return dataset
    
if __name__ == '__main__':
   # dataset_HP=load_HP_dataset()
    dataset_lfw=load_lfw_dataset()

    print dataset_HP.size
