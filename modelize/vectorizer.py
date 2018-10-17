import nltk
from gensim.models import fasttext

from utils.interfaces import NltkInterface


class SentenceGen:
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
                yield [x.lower() for x in tokens if NltkInterface.reg_exp.fullmatch(x)]


class Vectorizer:
    def __init__(self, csvreader):
        self._sentence_gen = SentenceGen(csvreader)

    def result(self):
        return fasttext.FastText(sentences=self._sentence_gen).wv
