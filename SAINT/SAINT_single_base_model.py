#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# # %tensorflow_version 1.x
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


# In[ ]:


# import tensorflow as tf
# if tf.test.gpu_device_name():
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
# else:
#     print("Please install GPU version of TF")


from time import time
t_init = time()


# In[ ]:


import os
import numpy as np
# import subprocess
import gc
import config
from config import *


# In[ ]:


import numpy as np
import keras
import random
from keras import backend as K
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Input,SpatialDropout1D, Embedding, LSTM, Dense, merge, Convolution2D, Lambda, GRU, TimeDistributed, Reshape, Permute, Convolution1D, Masking, Bidirectional
from keras.optimizers import Adam
from keras.layers import concatenate
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras import optimizers, callbacks


# In[ ]:


import gc
gc.collect()


# In[ ]:


#https://github.com/asjchen/secondary-folding/blob/master/src/predictors.py
from keras.layers import BatchNormalization, Dropout

# def truncated_accuracy(y_true, y_predict):
#     mask = K.sum(y_true, axis=2)
#     y_pred_labels = K.cast(K.argmax(y_predict, axis=2), 'float32')
#     y_true_labels = K.cast(K.argmax(y_true, axis=2), 'float32')
#     is_same = K.cast(K.equal(
#         y_true_labels, y_pred_labels), 'float32')
#     num_same = K.sum(is_same * mask, axis=1)
#     lengths = K.sum(mask, axis=1)
#     return K.mean(num_same / lengths, axis=0)

def truncated_accuracy(y_true, y_predict):
    mask = K.sum(y_true, axis=2)
    y_pred_labels = K.cast(K.argmax(y_predict, axis=2), 'float32')
    y_true_labels = K.cast(K.argmax(y_true, axis=2), 'float32')
    is_same = K.cast(K.equal(
        y_true_labels, y_pred_labels), 'float32')
    num_same = K.sum(is_same * mask, axis=1)
    lengths = K.sum(mask, axis=1)
    # return K.cast(K.sum(num_same, axis=0) / K.sum(lengths, axis=0), 'float32')
    return Lambda(lambda x: x[0]/x[1])([K.sum(num_same, axis=0), K.sum(lengths, axis=0)])


# In[ ]:


import math
import keras.backend as K
from keras.layers import Layer
from keras.initializers import Ones, Zeros
from keras.layers import Layer
class LayerNormalization(Layer):
    def __init__(self, eps: float = 1e-5, **kwargs) -> None:
        self.eps = eps
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=Zeros(), trainable=True)
        super().build(input_shape)

    def call(self, x, **kwargs):
        u = K.mean(x, axis=-1, keepdims=True)
        s = K.mean(K.square(x - u), axis=-1, keepdims=True)
        z = (x - u) / K.sqrt(s + self.eps)
        return self.gamma * z + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'eps': self.eps,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


# In[ ]:


from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import keras

from keras.layers import add

class MyLayer(Layer):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self._x = K.variable(0.2)
        self._x._trainable = True
        self.trainable_weights = [self._x]

        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        A, B = x
        result = add([self._x*A ,(1-self._x)*B])
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0]


# In[ ]:


from keras.layers import RepeatVector, Multiply, Flatten, Dot, Softmax, Lambda
from keras import backend

def attention(activations):
#https://arxiv.org/pdf/1703.03130.pdf
    d_a = 10
    r = 1
    units = activations.shape[2]
    #print(activations.shape)
    attention = TimeDistributed(Dense(d_a, activation='tanh', use_bias=False))(activations)
    attention = Dropout(.5)(attention)
    #print(attention.shape)
    attention = TimeDistributed(Dense(r, activation='softmax', use_bias=False))(activations) 
    #print(attention.shape)
    attention = Flatten()(attention)
    #attention = Activation('softmax')(attention)
    attention = RepeatVector(units)(attention)
    #print(attention.shape)
    attention = Permute([2, 1])(attention)
    #print(attention.shape)

    # apply the attention
    sent_representation = Multiply()([activations, attention])
    #sent_representation = Lambda(lambda xin: K.dot(xin[0], xin[1]))([attention, activations])
    #print(sent_representation.shape)
    #sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
    #print(sent_representation.shape)
    return sent_representation


def shape_list(x):
#   https://github.com/Separius/BERT-keras/blob/master/transformer/funcs.py
    if backend.backend() != 'theano':
        tmp = backend.int_shape(x)
    else:
        tmp = x.shape
    tmp = list(tmp)
    tmp[0] = -1
    return tmp


from keras.layers import RepeatVector

def attention_scaled_dot(activations, attention_mask): #, length):
#https://arxiv.org/pdf/1706.03762.pdf
    units = int(activations.shape[2])
    words = int(activations.shape[1])
    _drop_rate_ = .1
    Q = TimeDistributed(Dense(units, activation=None, use_bias=False))(activations)
    Q = Dropout(_drop_rate_)(Q)
    K = TimeDistributed(Dense(units, activation=None, use_bias=False))(activations)
    K = Dropout(_drop_rate_)(K)
    V = TimeDistributed(Dense(units, activation=None, use_bias=False))(activations)
    V = Dropout(_drop_rate_)(V)
    #print(Q.shape)
    QK_T = Dot(axes=-1, normalize=False)([Q,K]) # list of two tensors
    """normalize: Whether to L2-normalize samples along the dot product axis before taking the dot product. If set to True, then the output of the dot product is the cosine proximity between the two samples."""
    QK_T = Lambda( lambda inp: inp[0]/ backend.sqrt(backend.cast(shape_list(inp[1])[-1], backend.floatx())))([QK_T, V])
    #print(QK_T.shape)
    
#     cropping = np.zeros(QK_T.shape[1])
#     cropping[length:] = (-10**6) * np.ones(int(QK_T.shape[1])-length)
#     QK_T = QK_T + cropping
    attention_mask__ = RepeatVector(int(QK_T.shape[1]))(attention_mask)
#     print(attention_mask__.shape)
    QK_T = Add()([QK_T, attention_mask__])
    QK_T = Softmax(axis=-1)(QK_T)
    QK_T = Dropout(_drop_rate_)(QK_T)
    #print(V.shape)
    V = Permute([2, 1])(V)
    #print(V.shape)
    V_prime = Dot(axes=-1, normalize=False)([QK_T,V]) # list of two tensors
    #print(V_prime.shape)
    return V_prime


# In[ ]:


#https://github.com/Separius/BERT-keras/blob/master/transformer/embedding.py
import keras
import numpy as np
from keras.layers import Embedding, Add, concatenate
def _get_pos_encoding_matrix(protein_len: int, d_emb: int) -> np.array:
    pos_enc = np.array(
        [[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] if pos != 0 else np.zeros(d_emb) for pos in
         range(protein_len)], dtype=np.float32)
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc

def embeddings(inputs):
    gene_ids, pos_ids = inputs
    gene_vocab_len = 22
    protein_len = 700
    output_dim = 50
    gene_embedding = Embedding(gene_vocab_len, output_dim, input_length=protein_len,
                                              name='GeneEmbedding')(gene_ids)

    pos_embedding = Embedding(protein_len, output_dim, trainable=False, input_length=protein_len,
                                              name='PositionEmbedding',
                                              weights=[_get_pos_encoding_matrix(protein_len, output_dim)])(pos_ids)

    summation = Add(name='AddEmbeddings')([Dropout(.1, name='EmbeddingDropOut1')(gene_embedding), 
                                           Dropout(.1, name='EmbeddingDropOut2')(pos_embedding)])
    
#     summation = concatenate([Dropout(.1, name='EmbeddingDropOut1')(gene_embedding), 
#                              Dropout(.1, name='EmbeddingDropOut2')(pos_embedding)])
    
    summation = LayerNormalization(1e-5)(summation)
    return summation
  
def gene_embeddings(gene_ids, output_dim=50):
    gene_vocab_len = 22
    protein_len = 700
    
    gene_emb = Dropout(.1)(Embedding(gene_vocab_len, output_dim, input_length=protein_len,
                               name='GeneEmbedding')(gene_ids))

    gene_emb = LayerNormalization(1e-5)(gene_emb)
    return gene_emb


def position_embedding(pos_ids, output_dim=50):
    #gene_vocab_len = 22
    protein_len = 700
    output_dim = int(output_dim)

    pos_emb = Dropout(.1)(Embedding(protein_len, output_dim, trainable=False, input_length=protein_len,
                        #name='PositionEmbedding',
                        weights=[_get_pos_encoding_matrix(protein_len, output_dim)])(pos_ids))
    
    pos_emb = LayerNormalization(1e-5)(pos_emb)
    return pos_emb


# In[ ]:


def attention_module(x, attention_mask, pos_ids=None, drop_rate=.1):
    original_dim = int(x.shape[-1])
    if pos_ids is not None:
        pos_embedding = position_embedding(pos_ids=pos_ids, output_dim=original_dim)
        #x = concatenate([x, pos_embedding])
        x = Add()([x, pos_embedding])
    att_layer = attention_scaled_dot(x, attention_mask)
    att_layer = Dropout(drop_rate)(att_layer)
    x = MyLayer()([att_layer, x])
    x = Dropout(drop_rate)(x)
    x = BatchNormalization()(x)
#     if False:
#         # reduce dim
#         x = Convolution1D(original_dim, 1, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
#         x = Dropout(drop_rate*2)(x)
#         x = BatchNormalization()(x)
    return x


# In[ ]:


def inceptionBlock(x):
    _drop_rate_ = .1
    x = BatchNormalization()(x)
    conv1_1 = Convolution1D(100, 1, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    conv1_1 = Dropout(_drop_rate_)(conv1_1) #https://www.quora.com/Can-l-combine-dropout-and-l2-regularization
    conv1_1 = BatchNormalization()(conv1_1)
    
    conv2_1 = Convolution1D(100, 1, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    conv2_1 = Dropout(_drop_rate_)(conv2_1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_2 = Convolution1D(100, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv2_1)
    conv2_2 = Dropout(_drop_rate_)(conv2_2)
    conv2_2 = BatchNormalization()(conv2_2)
    
    conv3_1 = Convolution1D(100, 1, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    conv3_1 = Dropout(_drop_rate_)(conv3_1)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_2 = Convolution1D(100, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv3_1)
    conv3_2 = Dropout(_drop_rate_)(conv3_2)
    conv3_2 = BatchNormalization()(conv3_2)
    conv3_3 = Convolution1D(100, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv3_2)
    conv3_3 = Dropout(_drop_rate_)(conv3_3)
    conv3_3 = BatchNormalization()(conv3_3)
    conv3_4 = Convolution1D(100, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv3_3)
    conv3_4 = Dropout(_drop_rate_)(conv3_4)
    conv3_4 = BatchNormalization()(conv3_4)
    
    concat = concatenate([conv1_1, conv2_2, conv3_4])
    concat = BatchNormalization()(concat)
    
    return concat


# In[ ]:


def deep3iBLock_with_attention(x, attention_mask, pos_ids=None):
    block1_1 = inceptionBlock(x)
    block1_1 = attention_module(block1_1, attention_mask, pos_ids)
    
    block2_1 = inceptionBlock(x)
    block2_2 = inceptionBlock(block2_1)
    block2_2 = attention_module(block2_2, attention_mask, pos_ids)
    
    block3_1 = inceptionBlock(x)
    block3_2 = inceptionBlock(block3_1)
    block3_3 = inceptionBlock(block3_2)
    block3_4 = inceptionBlock(block3_3)
    block3_4 = attention_module(block3_4, attention_mask, pos_ids)
    
    concat = concatenate([block1_1, block2_2, block3_4])
    concat = BatchNormalization()(concat)
    
    return concat


# In[ ]:


def get_model(num_feature):
#   pssm_input = Input(shape=(700,21,), name='pssm_input')
#   seq_input = Input(shape=(700,22,), name='seq_input')
  _drop_rate_ = .4
  main_input = Input(shape=(700,num_feature,), name='main_input')
  attention_mask = Input(shape=(700,), name='attention_mask')
  pos_ids = Input(batch_shape=(None,700), name='position_input', dtype='int32')
  
  #pos_emb = position_embedding(pos_ids, output_dim=50)
#   main_input = concatenate([seq_input, pssm_input])
  
  block1 = deep3iBLock_with_attention(main_input, attention_mask, pos_ids)
#   att_layer_4 = attention_scaled_dot(block1)
#   block1 = MyLayer()([att_layer_4 ,block1])
#   block1 = BatchNormalization()(block1)
  
  block2 = deep3iBLock_with_attention(block1, attention_mask, pos_ids)
  block2 = attention_module(block2, attention_mask, pos_ids)
  
  conv11 = Convolution1D(100, 11, activation='relu', padding='same', kernel_regularizer=l2(0.001))(block2)
#   conv11 = BatchNormalization()(conv11)
  conv11 = attention_module(conv11, attention_mask, pos_ids)

  dense1 = TimeDistributed(Dense(units=256,activation='relu'))(conv11)
  dense1 = Dropout(_drop_rate_)(dense1)
  dense1 = attention_module(dense1, attention_mask, pos_ids)
  
  main_output = TimeDistributed(Dense(units=8,activation='softmax', name='main_output'))(dense1)
  
  model = Model([main_input, attention_mask, pos_ids],main_output)
  return model


# In[ ]:


class StepDecay():
  def __init__(self, initAlpha=0.0005, factor=0.9, dropEvery=60, min_lr=0.00001):
    self.initAlpha = initAlpha
    self.factor = factor
    self.dropEvery = dropEvery
    self.min_lr = min_lr

  def __call__(self, epoch):
    exp = np.floor((epoch + 1) / self.dropEvery)
    alpha = self.initAlpha * (self.factor ** exp)
    if float(alpha) > self.min_lr:
        print('lr:', alpha)
        return float(alpha)
    else:
        print('lr:', self.min_lr)
        return self.min_lr
    
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr
    
lr_decay = callbacks.LearningRateScheduler(StepDecay(initAlpha=0.0005, factor=0.9, dropEvery=10, min_lr=0.00001))


# In[ ]:


from time import time
t0 = time()

from GenerateBaseFeature import generateBaseDataset
from GenerateLengthList import generateLengthlistAndAttentionMask
print(time()-t0)


t0 = time()
base_dataset = [None] # generateBaseDataset(inputlist,inputdir) # Dimension = N x 700 x 57
lengths = [None] # generateLengthlist(inputlist,inputdir)

if True:
    from threading import Thread

    t1 = Thread(target=generateBaseDataset, args=(base_dataset, inputlist, inputdir)) 
    t5 = Thread(target=generateLengthlistAndAttentionMask, args=(lengths, inputlist, inputdir))

    t1.start() 
    t5.start() 

    t1.join() 
    t5.join() 

    print("Done!")

elif False:
    generateBaseDataset(base_dataset, inputlist,inputdir) # Dimension = N x 700 x 57
    generateLengthlist(lengths, inputlist,inputdir)


# print(base_dataset[0].shape, lengths[0][1].shape, time()-t0)


# In[ ]:


features0 = base_dataset[0]
attention_mask = lengths[0][1]
pos_ids = np.array(range(700))
pos_ids = np.repeat([pos_ids], int(base_dataset[0].shape[0]), axis=0)

print('Dataset loading time:', time()-t0)


# In[ ]:


adam = Adam()
from keras import backend as K

t00 = time()

t0 = time()
model = get_model(num_feature=57)
print('Model_creation_time:', time()-t0)

t1 = time()
model.load_weights('./SAINT_win0_weights.h5')
print('Weights_loading_time:', time()-t1)

model.compile(optimizer=adam,
              loss='categorical_crossentropy', #https://github.com/LucaAngioloni/ProteinSecondaryStructure-CNN/blob/master/dataset.py
              sample_weight_mode='temporal',
              metrics=[truncated_accuracy, 'accuracy','mae'])

t2 = time()
ss8_win0 = model.predict([features0, attention_mask, pos_ids])
print('Inference_time:', time()-t2)

#tx = time()
#del model
#for i in range(5):
#    K.clear_session()
#    gc.collect()
#print('Garbage collection time:', time()-tx)


# In[ ]:

#del model

# In[ ]:


# t000 = time()
ss8_win0 = np.argmax(ss8_win0, axis=-1)


_structures_ = 'LBEGIHSTX' # 'X' represents NoSeq, will not be in the ss8_string

ss8_win0_string = ''

for i in range(ss8_win0.shape[0]):
    for j in range(int(lengths[0][0][i])):
        ss8_win0_string += _structures_[ss8_win0[i, j]]
    ss8_win0_string += '\n'

"""
with open('outputs/SAINT_cwin0_output_ss8_sequences.txt', 'w') as f:
    f.write(ss8_win0_string)
"""

ss8_win0_string = ss8_win0_string.split('\n')

def getProtlist(inputlist):
    with open(inputlist,'r') as f:
        protlist = f.read().split('\n')
    if '' in protlist:
        protlist.remove('') #if there is any blank line in input file
    return protlist
    
protlist = getProtlist(inputlist=config.inputlist)

for i, prot_name in enumerate(protlist)
    with open('outputs/cwin0/{}.SAINT_cwin0.ss8'.format(prot_name), 'w') as f:
        f.write(ss8_win0_string[i])


print('Total time for script (SAINT_single_base):', time() - t_init)


# In[ ]:




