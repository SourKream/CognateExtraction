#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import codecs
import sys
import urllib2
from bs4 import BeautifulSoup
import pickle

UTF8Writer = codecs.getwriter('utf8')
sys.stdout = UTF8Writer(sys.stdout)

ieLex = {}

counter = 0

for meaning in open('Meanings.txt','r'):

	print "Meanings Scrapped : ", counter, " / 225\r",
	sys.stdout.flush()
	counter += 1

	meaning = meaning.strip()
	url = "http://ielex.mpi.nl/meaning/" + meaning + "/languagelist/all/"
	html = urllib2.urlopen(url).read()
	soup = BeautifulSoup(html, 'html.parser')

	td = soup.find_all('td')[1:]

	data = []
	for i in range(len(td)/7):
		data.append([td[7*i], td[7*i+1], td[7*i+2], td[7*i+3], td[7*i+5]])

	for i in range(len(data)):
		for j in range(5):
			data[i][j] = data[i][j].text.strip()

	with codecs.open(meaning + '.txt', 'w',encoding='utf-8') as f:
		for i in range(len(data)):
			for j in range(5):
				f.write(data[i][j] + '\t')
			f.write('\n')

	ieLex[meaning] = data

print "Done                           "

with open('ieLex.pkl','w') as fil:
	pickle.dump(ieLex, fil)
