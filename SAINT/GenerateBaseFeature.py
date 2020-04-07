#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import pandas as pd
import pickle
import os 
import config
  
def obtain_feats(id,mu,std,input_dir=''):
    features =      [['.hhm',1,read_hhm],
                    ['.pssm',1,read_pssm],
                    ['.fasta',1,read_pccp],]
    with open(input_dir+os.sep+id+'.fasta','r') as f:
        seq = ''.join(f.read().splitlines()[1:]).rstrip()
    inputs_oneD = []
    for i in features:
        tmp_feats = i[2](input_dir+os.sep+id+i[0],seq)
        if i[1] == 1:
            inputs_oneD.append(tmp_feats)
    inputs_oneD = np.concatenate(inputs_oneD,1)
    inputs_oneD = (inputs_oneD-mu)/std
    return inputs_oneD

def getpickledata(picklefile):
    with open(picklefile,'rb') as f:
            normdic = pickle.load(f,encoding='latin1')
            normmu = normdic['mu1d'][:57]
            normstd = normdic['std1d'][:57]
    return normmu,normstd


def read_pssm(fname,seq):
    num_pssm_cols = 44
    pssm_col_names = [str(j) for j in range(num_pssm_cols)]
    with open(fname,'r') as f:
        tmp_pssm = pd.read_csv(f,delim_whitespace=True,names=pssm_col_names).dropna().values[:,2:22].astype(float)
    if tmp_pssm.shape[0] != len(seq):
        raise ValueError('PSSM file is in wrong format or incorrect!')
    return tmp_pssm

def read_hhm(fname,seq):
    num_hhm_cols = 22
    hhm_col_names = [str(j) for j in range(num_hhm_cols)]
    with open(fname,'r') as f:
        hhm = pd.read_csv(f,delim_whitespace=True,names=hhm_col_names)
    pos1 = (hhm['0']=='HMM').idxmax()+3
    num_cols = len(hhm.columns)
    hhm = hhm[pos1:-1].values[:,:num_hhm_cols].reshape([-1,44])
    hhm[hhm=='*']='9999'
    if hhm.shape[0] != len(seq):
        raise ValueError('HHM file is in wrong format or incorrect!')
    return hhm[:,2:-12].astype(float)

with open('aa_phy7','r') as f:
    pccp = f.read().splitlines()
    pccp = [i.split() for i in pccp]
    pccp_dic = {i[0]: np.array(i[1:]).astype(float) for i in pccp}
    
def read_pccp(fname,seq):
    pccp = np.array([pccp_dic[i] for i in seq])
    return pccp



def getProtlist(inputlist):
    with open(inputlist,'r') as f:
        protlist = f.read().split('\n')
    protlist.remove('') #if there is any blank line in input file
    return protlist


def getlength(name, inputdir):
    with open(inputdir+os.sep+name+'.fasta','r') as f:
        fasta = f.read().split('\n')
    return len(fasta[1])

def getNoGreater700(protlist,inputdir):
    cnt = 0
    for x in protlist:
        if getlength(x,inputdir) > 700:
            cnt += 1
    return cnt

# In[54]:


from time import time

def generateBaseDataset(_dataset_, inputlist='protlist.txt',inputdir='inputs',picklefile = '5norm_data.p'):
    t0 = time()
    normmu,normstd = getpickledata(picklefile)
    print(inputlist,inputdir)
    cnt = 0
    protlist = getProtlist(inputlist)
    inputdir = os.getcwd()+os.sep+inputdir
    over700 = getNoGreater700(protlist,inputdir)
    dataset = np.zeros((len(protlist)+over700,700,57))
    for x in protlist:
        input_oneD = obtain_feats(x,normmu,normstd,inputdir)
        if input_oneD.shape[0] > 700:   
            dataset[cnt] = input_oneD[:700]
            cnt += 1
            dataset[cnt] = input_oneD[-700:]
            cnt += 1 
        else: 
            dataset[cnt, :input_oneD.shape[0]] = input_oneD
            cnt += 1
    _dataset_[0] = dataset
    print('generateBaseDataset took:', time()-t0)

