#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import pickle
import os 

def read_spotcon(fname,seq,min_sep=3):
    seqlen = len(seq)
    features = np.zeros([seqlen,seqlen,1])
    tmp_feat = features
    with open(fname,'r') as f:
        protein_pred = pd.read_csv(f,delim_whitespace=True,names=['pos1','pos2','idk1','idk2','score'])
    protein_pred = protein_pred[protein_pred['pos1'].astype(str).str.isdigit()].dropna().values
    if protein_pred.shape[0] == 0:
        # Older files without the 0 and 8 columns, and starting from base 0
        with open(fname,'r') as f:
            protein_pred = pd.read_csv(f,delim_whitespace=True,names=['pos1','pos2','score'])
        protein_pred = protein_pred[protein_pred['pos1'].astype(str).str.isdigit()].dropna().values    
        pos1 = protein_pred[:,0].astype(int)
        pos2 = protein_pred[:,1].astype(int) 
    else:  
        pos1 = protein_pred[:,0].astype(int)-1
        pos2 = protein_pred[:,1].astype(int)-1
    score = protein_pred[:,-1:]
    features[pos1,pos2] = score
    features = features + np.transpose(features,[1,0,2]) + np.tril(np.triu(np.ones([seqlen,seqlen]),-min_sep+1),min_sep-1)[:,:,None]
    return features

def spotcon_window_feats(spotcon_image,window_size):
    protlen = spotcon_image.shape[0]
    feat_depth = spotcon_image.shape[2]
    window_size = int(window_size)
    resize = np.concatenate([np.zeros([window_size,protlen,feat_depth]),np.concatenate([spotcon_image,np.zeros([window_size,protlen,feat_depth])],0)],0)
    spotcon_array = np.concatenate([resize[i:(i+2*window_size+1),i,:feat_depth] for i in range(protlen)],1).T
    rm_inds = np.array([window_size + i for i in range(-2,3)])
    return np.delete(spotcon_array,rm_inds,axis=1)

def obtain_feats(id,mu,std,window_size,input_dir=''):
    with open(input_dir+os.sep+id+'.fasta','r') as f:
        seq = ''.join(f.read().splitlines()[1:]).rstrip()
    inputs_twoD = []
    tmp_feats = read_spotcon(input_dir+os.sep+id+'.spotcon',seq)
    inputs_twoD.append(tmp_feats)
    inputs_twoD = np.concatenate(inputs_twoD,2)
    if window_size > 0:
        tmp_window_feats = spotcon_window_feats(inputs_twoD,window_size)
        inputs_oneD = tmp_window_feats
   
    inputs_oneD = (inputs_oneD-mu)/std
    return inputs_oneD


def getpickledata(picklefile):
    with open(picklefile,'rb') as f:
            normdic = pickle.load(f,encoding='latin1')
            normmu = normdic['mu1d']
            normstd = normdic['std1d']
    window_size = (normmu.shape[0] - 57 + 4)/2 if normmu.shape[0] != 57 else 0
    normmu = normmu[-36:]
    normstd = normstd[-36:]
    return normmu,normstd,window_size

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


# In[19]:


def getProtlist(inputlist):
    with open(inputlist,'r') as f:
        protlist = f.read().split('\n')
    if '' in protlist:
        protlist.remove('') #if there is any blank line in input file
    return protlist


# In[20]:


from time import time

def generateDatasetWindow20(_dataset_, inputlist='protlist.txt',inputdir='inputs',picklefile = '0norm_data.p'):
    t0 = time()
    normmu,normstd,window_size = getpickledata(picklefile)
    cnt = 0
    protlist = getProtlist(inputlist)
    inputdir = os.getcwd()+os.sep+inputdir
    over700 = getNoGreater700(protlist,inputdir)
    dataset = np.zeros((len(protlist)+over700,700,36))
    for x in protlist:
        input_oneD = obtain_feats(x,normmu,normstd,window_size,inputdir)
        if input_oneD.shape[0] > 700:   
            dataset[cnt] = input_oneD[:700]
            cnt += 1
            dataset[cnt] = input_oneD[-700:]
            cnt += 1 
        else: 
            dataset[cnt, :input_oneD.shape[0]] = input_oneD
            cnt += 1
    _dataset_[0] = dataset
    print('generateDatasetWindow10 took:', time()-t0)


# In[ ]:




