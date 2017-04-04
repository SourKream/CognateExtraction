import sys
import pickle

data = {}
meaning = ''
CCN = 0
remove_first_line = True

for line in open('Data/Mayan_asjp.tsv','r'):
	if remove_first_line:
		remove_first_line = False
	else:
		line = line.strip('\n').split('\t')
		if len(line) == 0:
			continue
		if line[2] not in data:
			data[line[2]] = []
		data[line[2]].append([line[1], line[0], line[5], '', line[6]])

with open('DataPickles/Mayan_asjp.pkl','w') as f:
	pickle.dump(data, f)
