#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import sys
import codecs

UTF8Writer = codecs.getwriter('utf8')
sys.stdout = UTF8Writer(sys.stdout)

POS_list = ['N_NN', 'V_VM']
data = {}
for POS in POS_list:
	data[POS] = []

for fn in range(25):
	hin_sent = open('Data/hin_tourism_set'+str(fn+1)+'.txt').readlines()[1:]
	mar_sent = open('Data/mar_tourism_set'+str(fn+1)+'.txt').readlines()[1:]

	for i in range(len(hin_sent)):
		hin_sent[i] = hin_sent[i].split('\t')[1]
		hin_sent[i] = hin_sent[i].strip().decode('utf-8').split(' ')
		hin_dict = {}
		for j in range(len(hin_sent[i])):
			word = hin_sent[i][j].split('\\')
			if len(word) == 2:
				if word[1] not in hin_dict:			
					hin_dict[word[1]] = word[0]

		mar_sent[i] = mar_sent[i].split('\t')[1]
		mar_sent[i] = mar_sent[i].strip().decode('utf-8').split(' ')
		mar_dict = {}
		for j in range(len(mar_sent[i])):
			word = mar_sent[i][j].split('\\')
			if len(word) == 2:
				if word[1] not in mar_dict:			
					mar_dict[word[1]] = word[0]

		for POS in POS_list:
			if POS in mar_dict and POS in hin_dict:
				data[POS].append([hin_dict[POS], mar_dict[POS]])

## Translate to IPA

hindi2ipa = {}
for line in open('hindi2ipa_Pruned.txt'):
	line = line.decode('utf-8').strip().split('\t')
	hindi2ipa[line[0]] = line[1]

def toIPA(word, hindi2ipa):
	word = list(word)
	for i in range(len(word)):
		if word[i] in hindi2ipa:
			word[i] = hindi2ipa[word[i]]
		else:
			word[i] = ''
	return ''.join(word)

data_ipa = {}
for PoS in data:
	data_ipa[PoS] = []
	for word1, word2 in data[PoS]:
		data_ipa[PoS].append([toIPA(word1, hindi2ipa), toIPA(word2, hindi2ipa)])
