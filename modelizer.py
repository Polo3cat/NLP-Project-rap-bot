#! /bin/python3

import argparse
import pickle
from csv import DictReader

from modelize.counter import Counter
from modelize.grammer import Grammer
from modelize.vectorizer import Vectorizer
from utils.interfaces import NltkInterface
from modelize.predecessorer import Predecessorer
from modelize.tabler import Tabler

parser = argparse.ArgumentParser(prog="Modelizer")
parser.add_argument('-o', help="output file for model", required=True)
parser.add_argument('-i', help="input csv file", required=True)
parser.add_argument('-m', help="model to generate: table or vector", default='table')

args = parser.parse_args()

counter = Counter()
grammer = Grammer()

with open(args.i, newline='') as i:
    csvreader = DictReader(i, dialect='unix')
    for song in csvreader:
        lyrics = song['lyrics']
        if not lyrics:
            continue
        lines = lyrics.splitlines()
        song_tags = []
        for line in lines:
            tokens = NltkInterface.tokenize(line)
            tagged_tokens = NltkInterface.tag(tokens)
            song_tags.append(NltkInterface.strip_words(tagged_tokens))
            counter.feed(tagged_tokens)
        grammer.feed(song_tags)

output = {'grammar': grammer.result(), 'vocabulary': counter.vocabulary(),}

if args.m == 'vector':
    predecessorer = Predecessorer(counter)
    output['predecessors'] = predecessorer.result()
    with open(args.i, newline='') as i:
        csvreader = DictReader(i, dialect='unix')
        vectorizer = Vectorizer(csvreader)
        output['ftvectors'] = vectorizer.result()
else:
    tabler = Tabler(counter)
    output['table'] = tabler.result()

with open(args.o, mode='bw') as f:
    pickle.dump(output, f)
