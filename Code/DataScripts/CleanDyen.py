## Preprocessing On Original File

# def removeBracketedString(a):
# 	s = ''
# 	i = 0
# 	while i < len(a):
# 		if a[i] != '(':
# 			s += a[i]
# 		else:
# 			while i < len(a) and a[i] != ')':
# 				i+=1
# 		i+=1
# 	return s.strip()

# fout = open('Data/Dyen_processed.txt','w')
# for line in open('Data/Dyen.txt','r'):
#     if len(line.strip()) > 0 and line.strip()[0] != 'c':
#         if line.strip()[0] == 'a' or line.strip()[0] == 'b':
#             fout.write(line)
#         else:
#             word = [removeBracketedString(item) for item in line[25:].strip().split(',')]
#             for w in word:
#                 if w != '':
#                     fout.write(line[:25] + w + '\n')
# fout.close()

import sys
import pickle

data = {}
meaning = ''
CCN = 0

for line in open('Data/Dyen_processed.txt','r'):
	data_line = line.strip().split()
	if len(data_line) == 0:
		continue
	elif data_line[0] == 'a':
		meaning = data_line[2]
		data[meaning] = []
	elif data_line[0] == 'b':
		CCN = int(data_line[1])
	else:
		data[meaning].append([data_line[1], data_line[2], line[25:].strip(), '', CCN])

with open('Pickles/DyenDataset.pkl','w') as f:
	pickle.dump(data, f)




