# filename = 'ieLex2016_asjp.tsv'
# filename = 'Austronesian_asjp.tsv'
filename = 'Mayan_asjp.tsv'

words = set([])
concepts_word = {}
langs_word = {}

for line in open(filename):
	line = line.strip().split('\t')
	word = line[5].lower()
	concept = line[2]
	lang = line[0]
	words.add(word)
	if word not in concepts_word:
	    concepts_word[word] = set([])
	concepts_word[word].add(concept)
	if word not in langs_word:
	    langs_word[word] = set([])
	langs_word[word].add(lang)

print "Analysizing Dataset : ", filename
print "Number of unique words ", len(words)

# ie_words = words
# au_words = words
my_words = words

A = []
for item in words:
     A.append(len(concepts_word[item]))

counter = {}
for val in A:
     if val not in counter:
          counter[val] = 0
     counter[val] += 1

print "Number of concepts per word ", np.mean(A)
print "Number of words with more than 1 concept ", len(words) - counter[1]

A = []
for item in words:
     A.append(len(langs_word[item]))

counter = {}
for val in A:
     if val not in counter:
          counter[val] = 0
     counter[val] += 1

print "Number of langs per word ", np.mean(A)
print "Number of words with more than 1 language ", len(words) - counter[1]
