import argparse
import pickle
from csv import DictReader

from counter import Counter
from grammer import Grammer
from interfaces import NltkInterface
from predecessorer import Predecessorer
from tabler import Tabler

parser = argparse.ArgumentParser(prog="Modelizer")
parser.add_argument('-o', help="output file for model", required=True)
parser.add_argument('-i', help="input csv file", required=True)

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

tabler = Tabler(counter)
predecessorer = Predecessorer(counter)

output = {'table': tabler.result(), 'grammar': grammer.result(),
          'predecessors': predecessorer.result(), 'vocabulary': counter.vocabulary()}
with open(args.o, mode='bw') as f:
    pickle.dump(output, f)
