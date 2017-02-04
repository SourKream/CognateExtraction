import pickle
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

with open('polyglot-en.pkl','r') as f:
	polyglot = pickle.load(f)

data = {}
for i in range(len(polyglot[0])):
	data[polyglot[0][i]] = polyglot[1][i]

error_count = 0
results = []
for line in open('combined.tab','r'):
	line = line.strip().split('\t')
	if line[0] not in data or line[1] not in data:
		error_count += 1
		continue
	else:
		dist = 10 - 5 * cosine(data[line[0]], data[line[1]])
		results.append([line[2], dist])

print spearmanr(np.matrix(results))


