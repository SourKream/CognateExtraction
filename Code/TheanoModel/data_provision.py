import os
import random
import logging
import numpy as np
from collections import OrderedDict, Counter

logger = logging.getLogger('root')

class DataProvision:

    def __init__(self, data_folder, tokenize_simple = False):

        self._word_pairs = OrderedDict()
        self._word1 = OrderedDict()
        self._word2 = OrderedDict()
        self._label = OrderedDict()
        self._pointer = OrderedDict()
        self._splits = ['Train', 'Test']
        self._tokenize_simple = tokenize_simple

        for split in self._splits:
            split_word_pairs = []
            split_label = []
            for line in open(os.path.join(data_folder, split) + '.txt'):
                line = line.strip().decode('utf-8').split('\t')
                split_word_pairs.append(line[:2])
                split_label.append(int(line[2]))
            self._word_pairs[split] = split_word_pairs
            self._label[split] = np.array(split_label)

        self.get_vocab()

        for split in self._splits:
            split_word1 = []
            split_word2 = []
            for word_pair in self._word_pairs[split]:
              split_word1.append(self.map_to_idx(self.tokenize(word_pair[0])))
              split_word2.append(self.map_to_idx(self.tokenize(word_pair[1])))
            self._word1[split] = np.array(split_word1)
            self._word2[split] = np.array(split_word2)
            self._pointer[split] = 0

        ## Shuffle Training Data
        idx = range(self._label['Train'].shape[0])
        random.shuffle(idx)
        self._word_pairs['Train'] = np.array(self._word_pairs['Train'])[idx]
        self._word1['Train'] = self._word1['Train'][idx]
        self._word2['Train'] = self._word2['Train'][idx]
        self._label['Train'] = self._label['Train'][idx]

        logger.info('Finished Loading Data')

    def map_to_idx(self, x):
        return [self._vocab[w] if w in self._vocab else self._vocab["unk"] for w in x]
    
    def get_vocab(self):
        vocab = Counter()
        for sample in self._word_pairs['Train']:
            tokens = self.tokenize(sample[0])
            tokens += self.tokenize(sample[1])
            vocab.update(tokens)
        tokens = ["unk", "delimiter", "pad_tok"] + [x for x, y in sorted(vocab.iteritems()) if y > 0]
        self._vocab = {y:x for x,y in enumerate(tokens)}

    def tokenize(self, word):
        if self._tokenize_simple:
            return list(word)

        arts = set([u'\u02b0', u'\u02b1', u'\u02b2', u'\u02b7', u'\u02b9', u'\u02c0', u'\u02c8', u'\u02cc',
                    u'\u02d0', u'\u02d1', u'\u02e0', u'\u0300', u'\u0301', u'\u0302', u'\u0303', u'\u0304',
                    u'\u0306', u'\u030a', u'\u030c', u'\u0311', u'\u031c', u'\u031d', u'\u031e', u'\u031f',
                    u'\u0320', u'\u0324', u'\u0325', u'\u0329', u'\u032a', u'\u032f', u'\u033b', u'\u035c',
                    u'\u0361'])
        segs = []
        curr = ''
        for i in range(len(word)):
            curr += word[i]
            if i+1 < len(word):
                if word[i+1] not in arts:
                    segs.append(curr)
                    curr = ''
        if curr != '':
            segs.append(curr)               
        return segs

    def get_size(self, partition):
        return self._label[partition].shape[0]

    def reset_pointer(self, partition):
        self._pointer[partition] = 0

    def iterate_batch(self, partition, batch_size):
        logger.debug('Begin to iterate batch for %s'%(partition))
        current = 0
        while current + batch_size <= self.get_size(partition):
            batch_word1 = self._word1[partition][current : current + batch_size]
            batch_word2 = self._word2[partition][current : current + batch_size]
            batch_label = self._label[partition][current : current + batch_size]

            current = current + batch_size
            logger.debug('Iterating batch at current: %d'%(current))
            yield batch_word1, batch_word2, batch_label

        if current != self.get_size(partition):
            batch_word1 = self._word1[partition][current :]
            batch_word2 = self._word2[partition][current :]
            batch_label = self._label[partition][current :]

            logger.debug('Finished iterating batch for %s'%(partition))
            yield batch_word1, batch_word2, batch_label

    def next_batch(self, partition, batch_size):
        if self._pointer[partition] + batch_size <= self.get_size(partition):
            batch_word1 = self._word1[partition][self._pointer[partition] : self._pointer[partition] + batch_size]
            batch_word2 = self._word2[partition][self._pointer[partition] : self._pointer[partition] + batch_size]
            batch_label = self._label[partition][self._pointer[partition] : self._pointer[partition] + batch_size]

            self._pointer[partition] = (self._pointer[partition] + batch_size) % self.get_size(partition)
            logger.debug('Next batch at pointer: %d'%(self._pointer[partition]))
            return batch_word1, batch_word2, batch_label
        else:
            logger.debug('New epoch of data iteration')
            next_pointer = (self._pointer[partition] + batch_size) % self.get_size(partition)

            batch_word1 = self._word1[partition][self._pointer[partition]:]
            batch_word1 = np.append(batch_word1, self._word1[partition][:next_pointer], axis = 0)
            batch_word2 = self._word2[partition][self._pointer[partition]:]
            batch_word2 = np.append(batch_word2, self._word2[partition][:next_pointer], axis = 0)
            batch_label = self._label[partition][self._pointer[partition]:]
            batch_label = np.append(batch_label, self._label[partition][:next_pointer], axis = 0)

            self._pointer[partition] = next_pointer
            logger.debug('Next batch at pointer: %d'%(next_pointer))
            return batch_word1, batch_word2, batch_label
