#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import argparse
import codecs
import logging
import numpy as np
import os
import pdb
import pickle
import sys
import theano

UTF8Writer = codecs.getwriter('utf8')
sys.stdout = UTF8Writer(sys.stdout)

np.random.seed(1337)
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2, activity_l2
from keras.callbacks import *
from keras.constraints import *
from keras.models import *
from keras.optimizers import *
from keras.utils.np_utils import to_categorical, accuracy
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

def get_params():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('-lstm', action="store", default=50, dest="lstm_units", type=int)
    parser.add_argument('-epochs', action="store", default=10, dest="epochs", type=int)
    parser.add_argument('-batch', action="store", default=128, dest="batch_size", type=int)
    parser.add_argument('-xmaxlen', action="store", default=12, dest="xmaxlen", type=int)
    parser.add_argument('-lr', action="store", default=0.001, dest="lr", type=float)
    parser.add_argument('-load', action="store", default=False, dest="load_save", type=bool)
    parser.add_argument('-l2', action="store", default=0.01, dest="l2", type=float)
    parser.add_argument('-dropout', action="store", default=0.1, dest="dropout", type=float)
    parser.add_argument('-local', action="store", default=False, dest="local", type=bool)
    parser.add_argument('-embd', action="store", default=80, dest='embd_size', type=int)
    parser.add_argument('-tkn_simple', action="store", default=False, dest='tokenize_simple', type=bool)
    opts = parser.parse_args(sys.argv[1:])
    print "lstm_units", opts.lstm_units
    print "epochs", opts.epochs
    print "batch_size", opts.batch_size
    print "xmaxlen", opts.xmaxlen
    print "regularization factor", opts.l2
    print "dropout", opts.dropout
    print "LR", opts.lr
    print "Embedding Size", opts.embd_size
    print "Tokenize Simple", opts.tokenize_simple
    return opts

def get_last(X):
    # get last element from time dimension
    return X[:, -1, :]

def build_model(opts, verbose=False):

    ## DYEN Model Part
    input_word_a_DY = Input(shape=(opts.xmaxlen,), dtype='int32', name="Input DYEN Word A")
    input_word_b_DY = Input(shape=(opts.xmaxlen,), dtype='int32', name="Input DYEN Word B")

    emb_layer_DY = Embedding(opts.vocab_size_DY, 
                            opts.embd_size,
                            W_constraint = unitnorm(),
                            input_length = opts.xmaxlen,
                            dropout = opts.dropout,
                            name = "Embedding Layer DYEN")    

    lstm_layer_DY = Bidirectional(LSTM(opts.lstm_units,
                                    return_sequences = True, 
                                    name="LSTM Layer DYEN"), name="Bidir LSTM Layer DYEN")
    lstm_dropout_layer_DY = Dropout(opts.dropout, name="LSTM Dropout Layer DYEN")

    word_a_DY = lstm_dropout_layer_DY (lstm_layer_DY (emb_layer_DY (input_word_a_DY)))
    word_b_DY = lstm_dropout_layer_DY (lstm_layer_DY (emb_layer_DY (input_word_b_DY)))

    ## IELex Model Part
    input_word_a = Input(shape=(opts.xmaxlen,), dtype='int32', name="Input Word A")
    input_word_b = Input(shape=(opts.xmaxlen,), dtype='int32', name="Input Word B")

    emb_layer = Embedding(opts.vocab_size, 
                            opts.embd_size,
##                            W_constraint = unitnorm(),
                            input_length = opts.xmaxlen,
                            dropout = opts.dropout,
                            name = "Embedding Layer")    

    lstm_layer = Bidirectional(LSTM(opts.lstm_units,
                                    return_sequences = True, 
                                    name="LSTM Layer"), name="Bidir LSTM Layer")
    lstm_dropout_layer = Dropout(opts.dropout, name="LSTM Dropout Layer")

    word_a = lstm_dropout_layer (lstm_layer (emb_layer (input_word_a)))
    word_b = lstm_dropout_layer (lstm_layer (emb_layer (input_word_b)))

    ## Attention Layer
    attention_layer = WbwAttentionLayer(return_att=True, name="Attention Layer")

    ## DYEN
    r_a_DY, _ = attention_layer ([word_a_DY, word_b_DY])
    r_b_DY, _ = attention_layer ([word_b_DY, word_a_DY])

    k = 2 * opts.lstm_units
    r_a_DY = Lambda(get_last, output_shape=(k,), name="r_a_n_DY")(r_a_DY)
    r_b_DY = Lambda(get_last, output_shape=(k,), name="r_b_n_DY")(r_b_DY)

    h_star_DY = Activation('tanh')(merge([r_a_DY, r_b_DY], mode='concat', concat_axis=1))

    output_layer_DY = Dense(1, activation='sigmoid', W_regularizer=l2(opts.l2), name="Output Layer DYEN")(h_star_DY)

    model_DY = Model(input = [input_word_a_DY, input_word_b_DY], output = output_layer_DY)
    model_DY.compile(loss='binary_crossentropy', optimizer=Adam(opts.lr))

    ## IELex
    r_a, alpha_a_b = attention_layer ([word_a, word_b])
    r_b, alpha_b_a = attention_layer ([word_b, word_a])

    k = 2 * opts.lstm_units
    r_a_n = Lambda(get_last, output_shape=(k,), name="r_a_n")(r_a)
    r_b_n = Lambda(get_last, output_shape=(k,), name="r_b_n")(r_b)

    h_star = Activation('tanh')(merge([r_a_n, r_b_n], mode='concat', concat_axis=1))

    output_layer = Dense(1, activation='sigmoid', W_regularizer=l2(opts.l2), name="Output Layer")(h_star)

    model = Model(input = [input_word_a, input_word_b], output = output_layer)
    model.compile(loss='binary_crossentropy', optimizer=Adam(opts.lr))

    print "DYEN MODEL"
    model_DY.summary()
    print "IELex MODEL"
    model.summary()
    print "Models Compiled"

    return model, model_DY

def compute_acc(X, Y, model, filename=None):
    scores = model.predict(X)
    plabels = np.round(scores)
    tlabels = np.matrix(Y).transpose()
    # print plabels
    # print tlabels
    p, r, f, _ = precision_recall_fscore_support(tlabels, plabels)

    # if filename != None:
    #     with open(filename, 'w') as f:
    #         for i, sample in enumerate(X):
    #             f.write(map_to_txt(sample, vocab)+"\t")
    #             f.write(str(plabels[i])+"\n")

    return p[1], r[1], f[1]

def getConfig(opts):
    conf = [opts.lstm_units, opts.embd_size, opts.vocab_size, opts.lr, opts.l2, opts.xmaxlen]
    return "_".join(map(lambda x: str(x), conf))


class Metrics(Callback):
    def __init__(self, data, labels, split="Training"):
        self.data = data
        self.labels = labels
        self.split = split

    def on_epoch_end(self, epochs, logs={}):
        pre, rec, fscore = compute_acc(self.data, self.labels, self.model)
        print "\n", self.split, " -> Precision: ", pre, "\t Recall: ", rec, "\t F-Score: ", fscore

class WeightSave(Callback):
    def __init__(self, path, config_str):
        self.path = path
        self.config_str = config_str

    def on_epoch_end(self,epochs, logs={}):
        self.model.save_weights(self.path + self.config_str +"_"+ str(epochs) +  ".weights") 

def PrepareData(dataPath, options):

    train = [line.strip().decode('utf-8').split('\t') for line in open(dataPath + 'Train.txt')]
    test = [line.strip().decode('utf-8').split('\t') for line in open(dataPath + 'Test.txt')]
    vocab = get_vocab(train, options.tokenize_simple)

    X_train, Y_train, labels_train = load_data(train, vocab, tokenize_simple = options.tokenize_simple)
    X_test,  Y_test,  labels_test  = load_data(test,  vocab, tokenize_simple = options.tokenize_simple)
   
    XMAXLEN = options.xmaxlen
    X_train = pad_sequences(X_train, maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'post')
    X_test  = pad_sequences(X_test,  maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'post')
    Y_train = pad_sequences(Y_train, maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'post')
    Y_test  = pad_sequences(Y_test,  maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'post')

    return X_train, Y_train, labels_train, X_test, Y_test, labels_test, vocab

if __name__ == "__main__":

    options = get_params()

    ## LOAD DATA
    X_train, Y_train, labels_train, X_test, Y_test, labels_test, vocab = PrepareData('./Data/IELex/', options)    
    X_train_DY, Y_train_DY, labels_train_DY, X_test_DY, Y_test_DY, labels_test_DY, vocab_DY = PrepareData('./Data/Dyen/', options)

    options.vocab_size = len(vocab)
    options.vocab_size_DY = len(vocab_DY)
    print "Vocab Size : ", len(vocab)

    ## LOAD MODEL   
    # options.load_save = True
    # MODEL_WGHT = './Models/CoAtt_Model_50_80_539_0.001_0.01_12_9.weights'

    if options.load_save and os.path.exists(MODEL_WGHT):
        print("Loading pre-trained model from ", MODEL_WGHT)
        model, model_DY = build_model(options)
        model.load_weights(MODEL_WGHT)

    else:
        print 'Building model'
        model, model_DY = build_model(options)

        print "Pretraining with DYEN Dataset"
        metrics_train = Metrics([X_train_DY, Y_train_DY], labels_train_DY, "Training")
        metrics_test = Metrics([X_test_DY, Y_test_DY], labels_test_DY, "Testing")

        history = model_DY.fit(x = [X_train_DY, Y_train_DY], 
                            y = labels_train_DY,
                            batch_size = options.batch_size,
                            nb_epoch = 10,
                            class_weight = {1:2.0, 0:1.0},
                            callbacks = [metrics_train, metrics_test])

        print 'Training IELex Model'
        ModelSaveDir = "./Models/Pret_CoAtt_Model_"
        save_weights = WeightSave(ModelSaveDir, getConfig(options))
        metrics_train = Metrics([X_train, Y_train], labels_train, "Training")
        metrics_test = Metrics([X_test, Y_test], labels_test, "Testing")

        history = model.fit(x = [X_train, Y_train], 
                            y = labels_train,
                            batch_size = options.batch_size,
                            nb_epoch = options.epochs,
                            class_weight = {1:2.0, 0:1.0},
                            callbacks = [save_weights, metrics_train, metrics_test])
