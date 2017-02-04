#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import sys
import codecs
import pickle
import numpy as np
import random
from math import sqrt
from scipy.sparse import csr_matrix as csr
from sklearn.metrics import confusion_matrix
from Utils import *
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer

UTF8Writer = codecs.getwriter('utf8')
sys.stdout = UTF8Writer(sys.stdout)

with open('Pickles/DyenDataset.pkl','r') as fil:
	Data = pickle.load(fil)

# with open('Pickles/ieLex2016.pkl','r') as fil:
# 	Data = pickle.load(fil)

with open('Pickles/tags.pkl','r') as fil:
	Tags = pickle.load(fil)

X = {}
Y = {}
for meaning in Data:
	X[meaning] = []
	Y[meaning] = []
	for i in range(len(Data[meaning])):
		for j in range(i+1, len(Data[meaning])):

			if Data[meaning][i][4] != '' and Data[meaning][j][4] != '':
				X[meaning].append((Data[meaning][i][2].lower(), Data[meaning][j][2].lower()))
				if Data[meaning][i][4] == Data[meaning][j][4]:
					Y[meaning].append(1)
				else:
					Y[meaning].append(-1)

words = set([])
for meaning in Data:
	for entry in Data[meaning]:
		words.add(entry[2].lower())

characters = set([])
for word in words:
	for char in word:
		characters.add(char)

V_set = set(['a','e','i','o','u','y'])
C_set = characters.difference(V_set)

CV_words = {}
for word in words:
	CV_word = ""
	for character in word:
		if character in V_set:
			CV_word += "V"
		else:
			CV_word += "C"
	CV_words[word] = CV_word

#########################################################
# def feature_for_word_pair(pair):
# 	global subseq_id
# 	global use_new_model
# 	global steal_ratio

# 	feature = {}

# 	for subseq in PHI[pair[0]]:
# 		if subseq in PHI[pair[1]]:
# 			feature[subseq_id[subseq]] = PHI[pair[0]][subseq] + PHI[pair[1]][subseq]

# 	extra_prob_mass = 0
# 	# steal_ratio = 0.35
# 	for subseq in feature:
# 		extra_prob_mass += steal_ratio * feature[subseq]
# 		feature[subseq] *= (1 - steal_ratio)

# 	weight_sum = 0
# 	non_common_subseq = []
# 	for subseq in PHI[pair[0]]:
# 		# if subseq not in PHI[pair[1]]:
# 		weight_sum += PHI[pair[0]][subseq]
# 	for subseq in PHI[pair[1]]:
# 		# if subseq not in PHI[pair[0]]:
# 		weight_sum += PHI[pair[1]][subseq]

# 	for subseq in PHI[pair[0]]:
# 		if subseq_id[subseq] not in feature:
# 			feature[subseq_id[subseq]] = 0
# 	for subseq in PHI[pair[1]]:
# 		if subseq_id[subseq] not in feature:
# 			feature[subseq_id[subseq]] = 0

# 	for subseq in PHI[pair[0]]:
# 		# if subseq not in PHI[pair[1]]:
# 		feature[subseq_id[subseq]] += PHI[pair[0]][subseq]*extra_prob_mass/weight_sum
# 	for subseq in PHI[pair[1]]:
# 		# if subseq not in PHI[pair[0]]:
# 		feature[subseq_id[subseq]] += PHI[pair[1]][subseq]*extra_prob_mass/weight_sum

# 	return feature

def feature_for_word_pair(pair):
	global subseq_id
	global use_new_model
	global steal_ratio

	feature_mult = {}
	feature_add = {}

	for subseq in PHI[pair[0]]:
		if subseq in PHI[pair[1]]:
			feature_mult[subseq_id[subseq]] = PHI[pair[0]][subseq] + PHI[pair[1]][subseq]

	for subseq in PHI[pair[0]]:
		feature_add[subseq_id[subseq]] = PHI[pair[0]][subseq]
	for subseq in PHI[pair[1]]:
		if subseq_id[subseq] in feature_add:
			feature_add[subseq_id[subseq]] += PHI[pair[1]][subseq]
		else:
			feature_add[subseq_id[subseq]] = PHI[pair[1]][subseq]


	feature = {}

	for subseq in feature_add:
		if subseq in feature_mult:
			feature[subseq] = steal_ratio*feature_add[subseq] + (1-steal_ratio)*feature_mult[subseq]
		else:
			feature[subseq] = steal_ratio*feature_add[subseq]

	return feature

#########################################################
## Parameters
max_subseq_length = 3
lamb = 0.7
use_new_model = True

# for lamb in [x/10.0 for x in range(1,10)]:
print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
print "LAMBDA : ", lamb, ", P : ", max_subseq_length, ", New Model : ", use_new_model

subsequences = set([])
for word in words:
	for subseq_length in range(2, max_subseq_length+1):
		for subseq in getSubsequences(word,subseq_length):
			subsequences.add(subseq)

for word in set(CV_words.values()):
	for subseq_length in range(2, max_subseq_length+1):
		for subseq in getSubsequences(word,subseq_length):
			subsequences.add(subseq)

subsequences = list(subsequences)
subseq_id = {y:x for x,y in enumerate(subsequences)}

PHI = {}
counter = 0
for word in words :

	print "Words Processed : ", counter, " / ", len(words), "\r",
	sys.stdout.flush()
	counter += 1

	feature_vec = {}
	for subseq_length in range(2, max_subseq_length+1):
		for subseq, span in getSubsequencesWithSpan(word, subseq_length):
			feature_vec[subseq] = feature_vec.setdefault(subseq,0) + pow(lamb, span[1]-span[0]+1)
		for subseq, span in getSubsequencesWithSpan(CV_words[word], subseq_length):
			feature_vec[subseq] = feature_vec.setdefault(subseq,0) + pow(lamb, span[1]-span[0]+1)

	sum_of_squares = 0
	for subseq in feature_vec:
		sum_of_squares += feature_vec[subseq]*feature_vec[subseq]
	sum_of_squares = sqrt(sum_of_squares)	
	for subseq in feature_vec:
		feature_vec[subseq] = feature_vec[subseq]/sum_of_squares

	PHI[word] = feature_vec

## Remove Duplicates
Combined = {}
for meaning in X:
	Combined[meaning] = set([])
	for i in range(len(X[meaning])):
		Combined[meaning].add((X[meaning][i], Y[meaning][i]))
	Combined[meaning] = list(Combined[meaning])
	random.shuffle(Combined[meaning])
	X[meaning] = []
	Y[meaning] = []
	for itemx, itemy in Combined[meaning]:
		X[meaning].append(itemx)
		Y[meaning].append(itemy)

for steal_ratio in [x/100.0 for x in range(0,110,10)]:
	steal_ratio = 0.5
	print "Steel Ratio : ", steal_ratio
	## Get Feature Vectors
	x = {}
	j = 0
	for meaning in X:
		x[meaning] = []
		for i in range(len(X[meaning])):
		 	x[meaning].append(feature_for_word_pair(X[meaning][i]))
		j += 1
		print "Progress : ", j, " / ", len(X), "\r",
		sys.stdout.flush()

	## K Fold Evaluation
	K = 5
	meanings = sorted(x.keys())

	avg_a = 0
	avg_p = 0
	avg_r = 0
	avg_f = 0

	for i in range(K):
		X_test = []
		x_test = []
		y_test = []
		X_train = []
		x_train = []
		y_train = []

		# # Fold On Meanings
		# for j in range(len(meanings)):
		# 	if j > (len(meanings) * i)/K and j < (len(meanings) * (i+1))/K :
		# 		X_test.extend(X[meanings[j]])			
		# 		x_test.extend(x[meanings[j]])
		# 		y_test.extend(Y[meanings[j]])

		# 	else:
		# 		X_train.extend(X[meanings[j]])
		# 		x_train.extend(x[meanings[j]])
		# 		y_train.extend(Y[meanings[j]])

		## Fold within Meanings and Ensure distinct words in test and train
		for j in range(len(meanings)):
			test_words = set([])

			for item in X[meanings[j]]:
				test_words.add(item[0])
				test_words.add(item[1])

			test_words = list(test_words)
			split = len(test_words)/K
			test_words = set(test_words[i*split:(i+1)*split])	
			
			for k in range(len(X[meanings[j]])):
				if X[meanings[j]][k][0] in test_words and X[meanings[j]][k][1] in test_words:
					X_test.append(X[meanings[j]][k])			
					x_test.append(x[meanings[j]][k])
					y_test.append(Y[meanings[j]][k])
				if X[meanings[j]][k][0] not in test_words and X[meanings[j]][k][1] not in test_words:				
					X_train.append(X[meanings[j]][k])
					x_train.append(x[meanings[j]][k])
					y_train.append(Y[meanings[j]][k])

		# ## Fold within Meanings
		# for j in range(len(meanings)):
		# 	split = int(len(x[meanings[j]])/K)
		# 	X_test.extend(X[meanings[j]][i*split:(i+1)*split])			
		# 	x_test.extend(x[meanings[j]][i*split:(i+1)*split])
		# 	y_test.extend(Y[meanings[j]][i*split:(i+1)*split])

		# 	X_train.extend(X[meanings[j]][:i*split] + X[meanings[j]][(i+1)*split:])
		# 	x_train.extend(x[meanings[j]][:i*split] + x[meanings[j]][(i+1)*split:])
		# 	y_train.extend(Y[meanings[j]][:i*split] + Y[meanings[j]][(i+1)*split:])

		# print "---------------------------------"
		# print "FOLD ", i+1, "\n"


		z = [(x_train[ii],y_train[ii]) for ii in range(len(x_train))]
		random.shuffle(z)
		x_train = [a for a,b in z]
		y_train = [b for a,b in z]


		print "Training on fold ", i+1
		print "Training set size : ", len(x_train)
		print "Positive Labels : ", (len(y_train) + sum(y_train)) / 2, "\n"

		v = DictVectorizer(sparse=True)
		x_train = v.fit_transform(x_train)
		print "Shape of Data : ",
		print x_train.shape
#		model = LinearSVC(dual=False, max_iter=1000, penalty='l2', C=35).fit(x_train, y_train)
		model = LinearSVC().fit(x_train, y_train)
		p_label = model.predict(x_train)
		showResults(y_train, p_label)

		print "\nTesting on fold ", i+1
		print "Testing set size : ", len(x_test)
		print "Positive Labels : ", (len(y_test) + sum(y_test)) / 2, "\n"

		p_label = model.predict(v.transform(x_test))
		showResults(y_test, p_label)
		a, p, r, f = getResults(y_test, p_label)
		avg_a += a
		avg_p += p
		avg_r += r
		avg_f += f
		print "---------------------------------"
		K = 1
		break

	# print "Average Results"
	# print "Accuracy  : ", avg_a/K, "%"
	# print "Precision : ", avg_p/K, "%"
	# print "Recall    : ", avg_r/K, "%"
	# print "F-Score   : ", avg_f/K
	# print "---------------------------------"

	print avg_a/K, "\t", avg_p/K, "\t", avg_r/K, "\t", avg_f/K
	print "---------------------------------"
