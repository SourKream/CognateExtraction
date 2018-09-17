#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

from collections import defaultdict
from keras.layers import *
from keras.models import Sequential, Model
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping
from keras import backend as K
from sklearn import metrics
from keras.regularizers import l2
import itertools as it
import numpy as np
import codecs, sys
import argparse
import logging
import numpy as np
import os
import pdb
import pickle
import theano
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from keras.callbacks import *
from keras.constraints import *
from keras.models import *
from keras.optimizers import *
from keras.utils.np_utils import to_categorical
from keras.layers.core import *
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import *
from keras.layers.wrappers import *
from keras.layers.normalization import BatchNormalization
from keras.layers import *
from sklearn.metrics import *
from Utils import *
from datetime import datetime
from matplotlib import cm
from AttentionLayer import *

UTF8Writer = codecs.getwriter('utf8')
sys.stdout = UTF8Writer(sys.stdout)

np.random.seed(1337)

def get_params():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('-lstm', action="store", default=40, dest="lstm_units", type=int)
    parser.add_argument('-epochs', action="store", default=20, dest="epochs", type=int)
    parser.add_argument('-batch', action="store", default=128, dest="batch_size", type=int)
    parser.add_argument('-xmaxlen', action="store", default=12, dest="xmaxlen", type=int)
    parser.add_argument('-lr', action="store", default=0.001, dest="lr", type=float)
    parser.add_argument('-l2', action="store", default=0.01, dest="l2", type=float)
    parser.add_argument('-dropout', action="store", default=0.1, dest="dropout", type=float)
    parser.add_argument('-embd', action="store", default=10, dest='embd_size', type=int)
    parser.add_argument('-tkn_simple', action="store", default=True, dest='tokenize_simple', type=bool)
    parser.add_argument('-langfeat', action="store", default=False, dest='use_lang_feat', type=bool)
    parser.add_argument('-conceptfeat', action="store", default=False, dest='use_concept_feat', type=bool)
    opts = parser.parse_args(sys.argv[1:])
    print "lstm_units", opts.lstm_units
    print "epochs", opts.epochs
    print "batch_size", opts.batch_size
    print "xmaxlen", opts.xmaxlen
    print "LR", opts.lr
    print "regularization factor", opts.l2
    print "dropout", opts.dropout
    print "Embedding Size", opts.embd_size
    print "Tokenize Simple", opts.tokenize_simple
    print "Language Features", opts.use_lang_feat
    print "Concept Features", opts.use_concept_feat
    return opts

options = get_params()

#####################################
## Model Properties

options.lstm_units = 75
options.embd_size = 30
file_path = './Models/RE_IPA_IELEX_DF1_CoAtt_Model_75_30_161_0.001_0.02_12_30.weights'
############################################

data_file = '../Code/DataPickles/CrossLanguage/IELex/LangInfo_DataFold1.pkl'
train_pairs, train_labels, test_pairs, test_labels, train_lang_pairs, test_lang_pairs = pickle.load(open(data_file))

languages = set([])
for lang_pair in train_lang_pairs:
    languages.update(lang_pair)
for lang_pair in test_lang_pairs:
    languages.update(lang_pair)
languages = list(languages)
unique_chars = set([])
for word_a, word_b in train_pairs:
    unique_chars.update(list(word_a))
    unique_chars.update(list(word_b))
for word_a, word_b in test_pairs:
    unique_chars.update(list(word_a))
    unique_chars.update(list(word_b))
unique_chars = list(unique_chars)

print len(unique_chars), " CHARACTERS"
# print unique_chars
n_dim = len(unique_chars)+1

print len(languages), " LANGUAGES"
# print languages
num_lang = len(languages)

############################################
## Prep data

train = []
test = []
for i in range(len(train_pairs)):
    train.append([train_pairs[i][0], train_pairs[i][1], train_labels[i]])
for i in range(len(test_pairs)):
    test.append([test_pairs[i][0], test_pairs[i][1], test_labels[i]])
vocab = get_vocab(train, options.tokenize_simple)

def load_data(train, vocab, labels = {'0':0,'1':1,0:0,1:1}, tokenize_simple = True):
    X,Y,Z = [],[],[]
    for p,h,l in train:
        p = map_to_idx(tokenize(p, tokenize_simple), vocab)
        h = map_to_idx(tokenize(h, tokenize_simple), vocab)
        if l in labels:         
            X += [p]
            Y += [h]
            Z += [labels[l]]
    return X,Y,Z

def load_test_data(data, vocab, maxlen = options.xmaxlen, tokenize_simple = True):
    X,Y = [],[]
    for p,h in data:
        p = map_to_idx(tokenize(p, tokenize_simple), vocab)
        h = map_to_idx(tokenize(h, tokenize_simple), vocab)
        X.append(p)
        Y.append(h)
    X = pad_sequences(X, maxlen=maxlen, value=vocab["pad_tok"], padding = 'post')
    Y = pad_sequences(Y, maxlen=maxlen, value=vocab["pad_tok"], padding = 'post')
    return X,Y

X_train, Y_train, labels_train = load_data(train, vocab)
X_test,  Y_test,  labels_test  = load_data(test,  vocab)

XMAXLEN = options.xmaxlen
X_train = pad_sequences(X_train, maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'post')
X_test  = pad_sequences(X_test,  maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'post')
Y_train = pad_sequences(Y_train, maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'post')
Y_test  = pad_sequences(Y_test,  maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'post')

options.vocab_size = len(vocab)
print "Vocab Size : ", len(vocab)

############################################
## Model

def get_last(X):
    # get last element from time dimension
    return X[:, -1, :]

def build_model(opts, verbose=False):

    input_word_a = Input(shape=(opts.xmaxlen,), dtype='int32', name="Input Word A")
    input_word_b = Input(shape=(opts.xmaxlen,), dtype='int32', name="Input Word B")

    emb_layer = Embedding(opts.vocab_size, 
                            opts.embd_size,
##                            W_constraint = unitnorm(),
                            input_length = opts.xmaxlen,
                            mask_zero = False,
                            name = "Embedding Layer")    
    emb_dropout_layer = SpatialDropout1D(opts.dropout)

    lstm_layer = Bidirectional(LSTM(opts.lstm_units,
                                    return_sequences = True, 
                                    name="LSTM Layer"), name="Bidir LSTM Layer")
    lstm_dropout_layer = SpatialDropout1D(opts.dropout, name="LSTM Dropout Layer")

    word_a = lstm_dropout_layer (lstm_layer (emb_dropout_layer (emb_layer (input_word_a))))
    word_b = lstm_dropout_layer (lstm_layer (emb_dropout_layer (emb_layer (input_word_b))))

    attention_layer = WbwAttentionLayer(return_att=True, name="Attention Layer")

    r_a, alpha_a_b = attention_layer ([word_a, word_b])
    r_b, alpha_b_a = attention_layer ([word_b, word_a])

    k = 2 * opts.lstm_units
    r_a_n = Lambda(get_last, output_shape=(k,), name="r_a_n")(r_a)
    r_b_n = Lambda(get_last, output_shape=(k,), name="r_b_n")(r_b)

    h_star = Activation('tanh')(concatenate([r_a_n, r_b_n], axis=1))
    
    if opts.use_lang_feat:
        input_lang_feat = Input(shape=(opts.num_lang,), dtype='int32', name="Input Lang Feat")
        h_star = concatenate([h_star, input_lang_feat], axis=1)
    
    if opts.use_concept_feat:
        input_concept_feat = Input(shape=(300,), name="Input Concept Feat")
        h_star = concatenate([h_star, input_concept_feat], axis=1)

    h_star = Dense(20, activation='tanh', kernel_regularizer=l2(opts.l2), name="Hidden Layer")(h_star)
    output_layer = Dense(1, activation='sigmoid', kernel_regularizer=l2(opts.l2), name="Output Layer")(h_star)

    if opts.use_lang_feat:
        model = Model(inputs = [input_word_a, input_word_b, input_lang_feat], outputs = output_layer)
    elif opts.use_concept_feat:
        model = Model(inputs = [input_word_a, input_word_b, input_concept_feat], outputs = output_layer)
    else:
        model = Model(inputs = [input_word_a, input_word_b], outputs = output_layer)

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(opts.lr))
    print "Model Compiled"

    att_model = Model(inputs = [input_word_a, input_word_b], outputs = alpha_a_b)

    return model, att_model
    # return model

def getConfig(opts):
    conf = [opts.lstm_units, opts.embd_size, opts.vocab_size, opts.lr, opts.l2, opts.xmaxlen]
    conf = "_".join(map(lambda x: str(x), conf))
    if opts.use_concept_feat:
        conf += '_ConceptFeat'
    if opts.use_lang_feat:
        conf += '_LangFeat'
    return conf

def compute_acc(X, Y, model, filename=None):
    scores = model.predict(X)
    plabels = np.round(scores)
    tlabels = np.matrix(Y).transpose()
    p, r, f, _ = precision_recall_fscore_support(tlabels, plabels)
    precision, recall, thresholds = precision_recall_curve(tlabels, scores)

    return p[1], r[1], f[1], auc(recall, precision)

#####################################
## Load Model

print 'Building model'
model, attention_model = build_model(options)

Weights = pickle.load(open(file_path))
model.set_weights(Weights)

#####################################
#####################################
#####################################/