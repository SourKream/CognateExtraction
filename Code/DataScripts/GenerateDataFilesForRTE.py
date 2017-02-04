#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import sys
import codecs
import pickle
import numpy as np

UTF8Writer = codecs.getwriter('utf8')
sys.stdout = UTF8Writer(sys.stdout)

#dataFile = 'DataPickles/Dyen/DataFold1.pkl'
dataFile = 'DataPickles/IELex/DataFold1.pkl'

X_train, y_train, X_test, y_test = pickle.load(open(dataFile,'r'))

with open('train.txt', 'w') as f:
	for i in range(len(X_train)):
		item = X_train[i]
		f.write(item[0])
#		f.write(' '.join(list(item[0].decode('utf-8'))).encode('utf-8'))
		f.write('\t')
		f.write(item[1])
#		f.write(' '.join(list(item[1].decode('utf-8'))).encode('utf-8'))
		f.write('\t')
		f.write(str(y_train[i]))
		f.write('\n')

with open('test.txt', 'w') as f:
	for i in range(len(X_test)):
		item = X_test[i]
		f.write(item[0])
#		f.write(' '.join(list(item[0].decode('utf-8'))).encode('utf-8'))
		f.write('\t')
		f.write(item[1])
#		f.write(' '.join(list(item[1].decode('utf-8'))).encode('utf-8'))
		f.write('\t')
		f.write(str(y_test[i]))
		f.write('\n')
