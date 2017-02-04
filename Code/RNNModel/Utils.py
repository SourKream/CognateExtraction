from sklearn.metrics import *
import numpy as np

def tokenize(sent):
    return sent.split(' ')

def map_to_idx(x, vocab):
    return [vocab[w] if w in vocab else vocab["unk"] for w in x]

def map_to_txt(x, vocab):
    textify = map_to_idx(x, inverse_map(vocab))
    return ' '.join(textify)
    
def inverse_map(vocab):
    return {vocab[item]: item for item in vocab}

def concat_in_out(X, Y, vocab):
    numex = X.shape[0] # num examples
    glue = vocab["delimiter"]*np.ones(numex).reshape(numex,1)
    inp_train = np.concatenate((X,glue,Y),axis=1)
    return inp_train

def getResults (labels, predicted):
    p, r, f, _ = precision_recall_fscore_support(labels, predicted)
    return p[1]*100, r[1]*100, f[1]
