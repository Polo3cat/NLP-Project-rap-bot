import re
import nltk
from gensim.models import fasttext
from sys import argv
import csv
import pickle

source = argv[1]
destiny = argv[2]


class SentenceGen:
    reg = re.compile(r'\w+')

    def __init__(self, r):
        self.r = r

    def __iter__(self):
        for row in self.r:
            lyrics = row['lyrics']
            if not lyrics:
                continue
            lines = lyrics.splitlines()
            for l in lines:
                tokens = nltk.tokenize.word_tokenize(l)
                yield [x.lower() for x in tokens if self.reg.fullmatch(x)]


with open(source, newline='') as sourcefile:
    csvreader = csv.DictReader(sourcefile, dialect='unix')
    fast_text = fasttext.FastText(sentences=SentenceGen(csvreader))
with open(destiny, mode='bw') as d:
    pickle.dump(fast_text.wv, d)
