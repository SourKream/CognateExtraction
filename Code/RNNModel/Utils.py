from sklearn.metrics import *
from collections import Counter
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def tokenize(word, simple = False):
    if simple:
        return list(word)

    arts = set([u'\u02b0', u'\u02b1', u'\u02b2', u'\u02b7', u'\u02b9', u'\u02c0', u'\u02c8', u'\u02cc',
                u'\u02d0', u'\u02d1', u'\u02e0', u'\u0300', u'\u0301', u'\u0302', u'\u0303', u'\u0304',
                u'\u0306', u'\u030a', u'\u030c', u'\u0311', u'\u031c', u'\u031d', u'\u031e', u'\u031f',
                u'\u0320', u'\u0324', u'\u0325', u'\u0329', u'\u032a', u'\u032f', u'\u033b', u'\u035c',
                u'\u0361'])
    segs = []
    curr = ''
    for i in range(len(word)):
        curr += word[i]
        if i+1 < len(word):
            if word[i+1] not in arts:
                segs.append(curr)
                curr = ''
    if curr != '':
        segs.append(curr)               
    return segs

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

def load_data(train, vocab, labels = {'0':0,'1':1}, tokenize_simple = False):
    X,Y,Z = [],[],[]
    for p,h,l in train:
        p = map_to_idx(tokenize(p, tokenize_simple), vocab)
        h = map_to_idx(tokenize(h, tokenize_simple), vocab)
        if l in labels:         
            X += [p]
            Y += [h]
            Z += [labels[l]]
    return X,Y,Z

def get_vocab(data, tokenize_simple = False):
    vocab = Counter()
    for ex in data:
        tokens = tokenize(ex[0], tokenize_simple)
        tokens += tokenize(ex[1], tokenize_simple)
        vocab.update(tokens)
    tokens = ["unk", "delimiter", "pad_tok"] + [x for x, y in sorted(vocab.iteritems()) if y > 0]
    vocab = {y:x for x,y in enumerate(tokens)}
    return vocab

def plot_attention(matrix, title='Attention matrix', cmap=plt.cm.Blues, labels=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, cmap=cmap)
    plt.title(title)
    fig.colorbar(cax)
    if labels:
        ax.set_xticklabels([''] + labels[1])
        ax.set_yticklabels([''] + labels[0])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()

def getPRCurveKeras (x_test, y_test, model, label='DL Model'):
    p_proba = model.predict(x_test)
    precision, recall, thresholds = precision_recall_curve(y_test, p_proba)
    plt.plot(recall, precision, label='{0} (AUC = {1:0.2f})'.format(label, auc(recall, precision)))

def getROCCurveKeras (x_test, y_test, model, label='DL Model'):
    p_proba = model.predict(x_test)
    fpr, tpr, thresholds = roc_curve(y_test, p_proba)
    plt.plot(fpr, tpr, label='{0} (AUC = {1:0.2f})'.format(label, auc(fpr, tpr)))
