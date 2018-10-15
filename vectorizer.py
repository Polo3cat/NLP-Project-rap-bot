from gensim.models import fasttext
from sys import argv
import csv
import pickle

source = argv[1]
destiny = argv[2]


class SentenceGen:
    def __init__(self, csvreader):
        self.csvreader = csvreader

    def __iter__(self):
        for row in self.csvreader:
            lyrics = row['lyrics']
            if not lyrics:
                continue
            lines = lyrics.splitlines()
            for l in lines:
                yield l


with open(source, newline='') as sourcefile:
    csvreader = csv.DictReader(sourcefile, dialect='unix')
    fast_text = fasttext.FastText(sentences=SentenceGen(csvreader))
with open(destiny, mode='bw') as d:
    pickle.dump(fast_text.wv, d)
