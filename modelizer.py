#! /bin/python3

import argparse
import pickle
from csv import DictReader

from modelize.counter import Counter
from modelize.grammer import Grammer
from modelize.vectorizer import Vectorizer
from modelize.predecessorer import Predecessorer
from modelize.tabler import Tabler
import utils.interfaces


parser = argparse.ArgumentParser(prog="Modelizer")
parser.add_argument('-o', help="output file for model", required=True)
parser.add_argument('-i', help="input csv file", required=True)
parser.add_argument('-m', help="model to generate: table or vector", default='table', choices=['table', 'vector'])
parser.add_argument('-g', help="use a more general grammar", action='store_true')

args = parser.parse_args()

nltki = utils.interfaces.NltkInterface
if args.g:
    nltki = utils.interfaces.NltkInterfaceGeneralised

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
            tokens = nltki.tokenize(line)
            tagged_tokens = nltki.tag(tokens)
            song_tags.append(nltki.strip_words(tagged_tokens))
            counter.feed(tagged_tokens)
        grammer.feed(song_tags)

output = {'grammar': grammer.result(), 'vocabulary': counter.vocabulary(), }

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
