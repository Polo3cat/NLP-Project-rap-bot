#!/bin/python3

import sys
import nltk
import csv
import re
import numpy as np
import pickle


source = sys.argv[1]
savefile_table = sys.argv[2]
savefile_hash = sys.argv[3]
types_count = {}
bigrams_count = {}
conditional_probabilities = {}
type_pos_map = {}
pos_types = {}
total_w = 0

with open(source, newline='') as sourcefile:
    csvreader = csv.DictReader(sourcefile, dialect='unix')
    for row in csvreader:
        lyrics = row['lyrics']
        if not lyrics:
            continue
        sentences = lyrics.splitlines()
        for sentence in sentences:
            tokens = nltk.tokenize.word_tokenize(sentence)
            tokens = [x.lower() for x in tokens if re.fullmatch(r'\w+', x)]
            tagged_tokens = nltk.pos_tag(tokens)
            tagged_tokens.reverse()
            for t in tagged_tokens:
                types_count[t] = 1 + types_count.get(t, 0)
                pos_types[t[1]] = pos_types.get(t[1], set()) | {t[0]}
                total_w += 1
            for b in zip(tagged_tokens, tagged_tokens[1:]):
                bigrams_count[b] = 1 + bigrams_count.get(b, 0)

sh_count = 0
silly_hash = {}
silly_vector = [None]*len(types_count)
prob_table = np.empty((len(types_count), len(types_count)))

# P(w1|w0) = C(w0 w1) / C(w0)
for b in bigrams_count:
    w0 = b[0]
    w1 = b[1]
    if w0 not in silly_hash:
        silly_hash[w0] = sh_count
        silly_vector[sh_count] = w0
        sh_count += 1
    if w1 not in silly_hash:
        silly_hash[w1] = sh_count
        silly_vector[sh_count] = w1
        sh_count += 1
    row = silly_hash[w0]
    col = silly_hash[w1]
    prob_table[row, col] = (bigrams_count[b] + 1) / (types_count[w0] + total_w)

for i in range(len(prob_table)):
    for j in range(len(prob_table)):
        if not prob_table[i, j]:
            prob_table[i, j] = 1 / total_w

for w in types_count:
    if w not in silly_hash:
        silly_hash[w] = sh_count
        silly_vector[sh_count] = w
        sh_count += 1
print(sys.getsizeof(prob_table))
with open(savefile_table, mode='bw') as f:
    np.save(f, prob_table, allow_pickle=False)
with open(savefile_hash, mode='bw') as f:
    pickle.dump({'hash': silly_hash, 'vector': silly_vector}, f)
