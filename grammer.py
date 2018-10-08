import sys
import nltk
import csv
import re


source = sys.argv[1]
types_count = {}
bigrams_count = {}
conditional_probability_type = {}
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
            tagged_tokens= nltk.pos_tag(tokens)
            tagged_tokens.reverse()
            for t in tagged_tokens:
                types_count[t] = 1 + types_count.get(t, 0)
            for b in zip(tagged_tokens, tagged_tokens[1:]):
                bigrams_count[b] = 1 + bigrams_count.get(b, 0)

    for b in bigrams_count:
        conditional_probability_type[b] = bigrams_count[b] / types_count[b[1]]

for k, probability in conditional_probability_type.items():
    w_type = k[0]
    condition = k[1][0]
    condition_pos = k[1][1]
    if w_type not in type_pos_map:
        type_pos_map[w_type] = {condition_pos: (condition, probability)}
    elif condition_pos not in type_pos_map[w_type] or type_pos_map[w_type][condition_pos][1] > probability:
        type_pos_map[w_type][condition_pos] = (condition, probability)
        
for k, v in type_pos_map.items():
    print(k, v)
