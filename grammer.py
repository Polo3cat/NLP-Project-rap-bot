import sys
import nltk
import csv
import re

source = sys.argv[1]
bigrams = {}


with open(source, newline='') as sourcefile:
    csvreader = csv.DictReader(sourcefile, dialect='unix')
    for row in csvreader:
        lyrics = row['lyrics']
        if not lyrics:
            continue
        sentences = lyrics.splitlines()
        for sentence in sentences:
            tokens = nltk.tokenize.word_tokenize(sentence)
            tokens = [x for x in tokens if re.match(r'\w', x)]
            tokens.reverse()
            for b in zip(tokens, tokens[1:]):
                if b in bigrams:
                    bigrams[b] += 1
                else:
                    bigrams[b] = 1

