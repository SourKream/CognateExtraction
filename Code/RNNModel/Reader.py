from Utils import *
from collections import Counter

def load_data(train, vocab, labels = {'0':0,'1':1}):
    X,Y,Z = [],[],[]
    for p,h,l in train:
        p = map_to_idx(tokenize(p), vocab)
        h = map_to_idx(tokenize(h), vocab)
        if l in labels:         
            X += [p]
            Y += [h]
            Z += [labels[l]]
    return X,Y,Z

def get_vocab(data):
    vocab = Counter()
    for ex in data:
        tokens = tokenize(ex[0])
        tokens += tokenize(ex[1])
        vocab.update(tokens)
    lst = ["unk", "delimiter", "pad_tok"] + [ x for x, y in vocab.iteritems() if y > 0]
    vocab = dict([ (y,x) for x,y in enumerate(lst) ])
    return vocab

