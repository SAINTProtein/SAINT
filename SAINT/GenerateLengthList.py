#!/usr/bin/env python
# coding: utf-8

# In[17]:


import os
import numpy as np
import config

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

def getProtlist(inputlist):
    with open(inputlist,'r') as f:
        protlist = f.read().split('\n')
    if '' in protlist:
        protlist.remove('') #if there is any blank line in input file
    return protlist

def generateLengthlistAndAttentionMask(_dataset_, inputlist='protlist.txt',inputdir='inputs'):
    dict = {}
    cnt = 0
    protlist = getProtlist(inputlist)
    lenlist = []
    over700 = getNoGreater700(protlist,inputdir)
    attention_mask = np.zeros((len(protlist)+over700, 700))
    inputdir = os.getcwd()+os.sep+inputdir
    for x in protlist:
        length = getlength(x,inputdir)
        if length > 700:
            dict[cnt] = [2,length]
            lenlist.append(700)
            lenlist.append(700)
            cnt += 2
        else:
            lenlist.append(length)
            attention_mask[cnt, int(length):] = -1000000
            cnt += 1 
    _dataset_[0] = (np.array(lenlist), attention_mask, dict)
    #return np.array(lenlist),dict
        


# In[18]:

