#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import sys
import codecs
import pickle
import numpy as np
import random

UTF8Writer = codecs.getwriter('utf8')
sys.stdout = UTF8Writer(sys.stdout)

#dataFile = 'DataPickles/DyenDataset.pkl'
dataFile = 'DataPickles/ieLex2016.pkl'

Data = pickle.load(open(dataFile,'r'))

numFolds = 5
words = {}

for meaning in Data:
	random.shuffle(Data[meaning])
	words[meaning] = []
	for word in Data[meaning]:
		if word[4] != '':
			words[meaning].append((word[2], word[4]))

for fold in range(numFolds):

	X_train = []
	X_test = []
	y_train = []
	y_test = []


	for meaning in words:

		numWordsInMeaning = len(words[meaning])
		numWordsPerFold = numWordsInMeaning/numFolds

		testWords = set(words[meaning][fold*numWordsPerFold:(fold+1)*numWordsPerFold])
		trainWords = []
		for word in words[meaning]:
			if word not in testWords:
				trainWords.append(word)
		testWords = list(testWords)

		for i, word_i in enumerate(trainWords):
			for j, word_j in enumerate(trainWords[i+1:]):
				X_train.append((word_i[0].lower(), word_j[0].lower()))
				if word_i[1] == word_j[1]:
					y_train.append(1)
				else:
					y_train.append(0)

		for i, word_i in enumerate(testWords):
			for j, word_j in enumerate(testWords[i+1:]):
				X_test.append((word_i[0].lower(), word_j[0].lower()))
				if word_i[1] == word_j[1]:
					y_test.append(1)
				else:
					y_test.append(0)


	pickle.dump([X_train, y_train, X_test, y_test], open('DataFold'+str(fold+1)+'.pkl', 'w'))	
