#!/bin/python3

import sys
import nltk
import csv
import re
import pickle


source = sys.argv[1]
savefile = sys.argv[2]
types_count = {}
bigrams_count = {}
conditional_probabilities = {}
type_pos_map = {}

with open(source, newline='') as sourcefile:
    csvreader = csv.DictReader(sourcefile, dialect='unix')
    for row in csvreader:
        lyrics = row['lyrics']
        if not lyrics:
            continue
        sentences = lyrics.splitlines()
        for sentence in sentences:
            tokens = nltk.tokenize.word_tokenize(sentence)
            tokens = [x.lower() for x in tokens if re.match(r'\w', x)]
            tagged_tokens = nltk.pos_tag(tokens)
            tagged_tokens.reverse()
            for t in tagged_tokens:
                types_count[t] = 1 + types_count.get(t, 0)
            for b in zip(tagged_tokens, tagged_tokens[1:]):
                bigrams_count[b] = 1 + bigrams_count.get(b, 0)

# P(w1|w0) = C(w0 w1) / C(w0)
for b in bigrams_count:
    conditional_probabilities[b] = bigrams_count[b] / types_count[b[0]]

# For each w0 keep the highest P for each PoS tag
for bigram, probability in conditional_probabilities.items():
    w0 = bigram[0]
    w1_type = bigram[1][0]
    w1_pos = bigram[1][1]
    if w0 not in type_pos_map:
        type_pos_map[w0] = {w1_pos: (w1_type, probability)}
    elif w1_pos not in type_pos_map[w0] or type_pos_map[w0][w1_pos][1] > probability:
        type_pos_map[w0][w1_pos] = (w1_type, probability)

# strip away probabilities as they won't be used anymore
w0_pos_to_pos_w1 = {}
for w0, words1 in type_pos_map.items():
    aux_dict = {}
    for k, v in type_pos_map[w0].items():
        aux_dict[k] = v[0]
    w0_pos_to_pos_w1[w0] = aux_dict

with open(savefile, mode='bw') as f:
    pickle.dump(w0_pos_to_pos_w1, f)
