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
# dataFile = 'DataPickles/Austronesian_asjp.pkl'
# dataFile = 'DataPickles/ieLex2016_asjp.pkl'
# dataFile = 'DataPickles/Mayan_asjp.pkl'

Data = pickle.load(open(dataFile,'r'))

numFolds = 5
words = {}
languages = set([])

for meaning in Data:
	random.shuffle(Data[meaning])
	words[meaning] = []
	for word in Data[meaning]:
		if word[4] != '':
			words[meaning].append((word[2], word[4], word[1]))
			languages.add(word[1])

languages = list(languages)
random.shuffle(languages)

for fold in range(numFolds):

	X_train = []
	X_test = []
	y_train = []
	y_test = []
	lang_train = []
	lang_test = []

	numLanguagesPerFold = len(languages)/numFolds
	testLanguages = set(languages[fold*numLanguagesPerFold:(fold+1)*numLanguagesPerFold])	

	for meaning in words:

		trainWords = []
		testWords = []
		for word in words[meaning]:
			if word[2] not in testLanguages:
				trainWords.append(word)
			else:
				testWords.append(word)

		for i, word_i in enumerate(trainWords):
			for j, word_j in enumerate(trainWords[i+1:]):
				X_train.append((word_i[0].lower(), word_j[0].lower()))
				lang_train.append((word_i[2], word_j[2]))
				if word_i[1] == word_j[1]:
					y_train.append(1)
				else:
					y_train.append(0)

		for i, word_i in enumerate(testWords):
			for j, word_j in enumerate(testWords[i+1:]):
				X_test.append((word_i[0].lower(), word_j[0].lower()))
				lang_test.append((word_i[2], word_j[2]))
				if word_i[1] == word_j[1]:
					y_test.append(1)
				else:
					y_test.append(0)


	pickle.dump([X_train, y_train, X_test, y_test], open('DataFold'+str(fold+1)+'.pkl', 'w'))	
	pickle.dump([X_train, y_train, X_test, y_test, lang_train, lang_test], open('LangInfo_DataFold'+str(fold+1)+'.pkl','w'))


