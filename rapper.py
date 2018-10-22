import random

import numpy as np
import requests

from utils.interfaces import NltkInterface
from utils.interfaces import NltkInterfaceGeneralised


class NoRhyme(BaseException):
    pass


class Poet:
    def __init__(self, nltki=NltkInterface):
        self._uris = ["https://api.datamuse.com/words?rel_rhy={}", "http://rhymebrain.com/talk?function=getRhymes&word={}"]
        self._nltki = nltki

    @classmethod
    def _leven_distance(cls, str1, str2):
        d = dict()
        for i in range(len(str1) + 1):
            d[i] = dict()
            d[i][0] = i
        for i in range(len(str2) + 1):
            d[0][i] = i
        for i in range(1, len(str1) + 1):
            for j in range(1, len(str2) + 1):
                d[i][j] = min(d[i][j - 1] + 1, d[i - 1][j] + 1, d[i - 1][j - 1] + (not str1[i - 1] == str2[j - 1]))
        return d[len(str1)][len(str2)]

    def rhyme(self, word, pos):
        for uri in self._uris:
            url = uri.format(word)
            r = requests.get(url)
            json = r.json()
            for j in json:
                word = j['word']
                p = self._nltki.tag_word(word)
                if p == pos:
                    return word
        raise NoRhyme

    def rhyme_in(self, word, pos, bag):
        for uri in self._uris:
            url = uri.format(word)
            r = requests.get(url)
            json = r.json()
            for j in json:
                word = j['word']
                p = self._nltki.tag_word(word)
                if p == pos and word in bag:
                    return word
        raise NoRhyme

    def rhyme_random(self, word, pos):
        candidates = []
        for uri in self._uris:
            url = uri.format(word)
            r = requests.get(url)
            json = r.json()
            for j in json:
                word = j['word']
                p = self._nltki.tag_word(word)
                if p == pos:
                    candidates.append(word)
        if not candidates:
            raise NoRhyme
        return random.choice(candidates[:min(len(candidates), 5)])

    @classmethod
    def find_closest(cls, elem, bag):
        best_leven = 1000
        best_elem = elem
        for e in bag:
            leven = cls._leven_distance(elem, e)
            if leven < best_leven:
                best_leven = leven
                best_elem = e
        return best_elem

    def tokenize(self, sentence):
        return self._nltki.tokenize(sentence)

    def just_tags(self, tokens):
        return self._nltki.just_tags(tokens)

    def rhyme_best(self, word):
        for uri in self._uris:
            url = uri.format(word)
            r = requests.get(url)
            json = r.json()
            for j in json:
                word = j['word']
                p = self._nltki.tag_word(word)
                return word, p
        raise NoRhyme

    @classmethod
    def find_closest_with(cls, elem, pos, bag):
        best_leven = 1000
        best_elem = elem
        for e in bag:
            if e:
                leven = cls._leven_distance(elem, e)
                if leven < best_leven and e[-1] == pos:
                    best_leven = leven
                    best_elem = e
        return best_elem


class MetaRapper(type):
    def __new__(mcs, name, bases, namespace):
        if '_answer' not in namespace:
            raise TypeError('_answer was not implemented in {}'.format(name))
        return super().__new__(mcs, name, bases, namespace)


class BaseRapper(metaclass=MetaRapper):
    """
    Base class to generate a Rapper. Child classes must implement _answer.
    """

    def __init__(self, grammar, vocabulary):
        """
        :type grammar: dict
        :type vocabulary: dict
        """
        self._grammar = grammar
        self._vocabulary = vocabulary
        self._poet = Poet()

    def rap(self, sentence):
        """
        :param sentence: a well formed sentence. Has word characters.
        :type sentence: str
        :return: a rhyming sentence
        :rtype str | None
        """
        tokens = self._poet.tokenize(sentence)
        og_struct = self._poet.just_tags(tokens)
        if og_struct in self._grammar:
            gram_struct = self._grammar[og_struct]
        else:
            gram_struct = self._grammar[Poet.find_closest(og_struct, self._grammar)]
        return self._answer(tokens, gram_struct)

    def _answer(self, tokens, gram_struct):
        raise NotImplementedError


class FastTextRapper(BaseRapper):
    """
    Rapper that uses fast-text to look for similar words when it can't find a predecessor
    """

    def __init__(self, grammar, predecessors, vocabulary, fast_text_keyed_vectors):
        """
        :type grammar: dict
        :type predecessors: dict
        :type vocabulary: dict
        :type fast_text_keyed_vectors: gensim.models.keyedvectors.FastTextKeyedVectors
        """
        self._fast_text_keyed_vectors = fast_text_keyed_vectors
        self._predecessors = predecessors
        super(FastTextRapper, self).__init__(grammar, vocabulary)

    def _answer(self, tokens, gram_struct):
        try:
            rhyme = self._poet.rhyme(tokens[-1], gram_struct[-1])
        except NoRhyme:
            return None
        answer = [rhyme]
        previous_w = rhyme
        previous_pos = gram_struct[-1]
        for tag in reversed(gram_struct[:-1]):
            if (previous_w, previous_pos) not in self._predecessors or \
                    tag not in self._predecessors[(previous_w, previous_pos)]:
                previous_w = self._find_similar(previous_w, tag)
            else:
                previous_w = self._predecessors[(previous_w, previous_pos)][tag]
            if not previous_w:
                return None
            answer.append(previous_w)
            previous_pos = tag
        return " ".join(reversed(answer)).capitalize()

    def _find_similar(self, w0, desired_pos):
        w1 = None
        best_similarity = -1.
        for w in self._vocabulary[desired_pos]:
            try:
                similarity = self._fast_text_keyed_vectors.similarity(w0, w)
                if best_similarity < similarity:
                    best_similarity = similarity
                    w1 = w
            except KeyError:
                continue
        return w1


class ExhaustiveRapper(BaseRapper):
    """
    Back tracking Rapper that uses a table containing all the bi-gram probabilities to look for predecessors.
    The matrix is stored as a csr (Compressed Sparse Row) matrix, so it is much more compact
    """
    def __init__(self, grammar, vocabulary, table):
        self._table = table
        super(ExhaustiveRapper, self).__init__(grammar, vocabulary)

    def _answer(self, tokens, gram_struct):
        try:
            rhyme = self._poet.rhyme_random(tokens[-1], gram_struct[-1])
        except NoRhyme:
            try:
                rhyme, pos = self._poet.rhyme_best(tokens[-1])
            except NoRhyme:
                return None
            gram_struct = Poet.find_closest_with(gram_struct, pos, self._grammar)
        answer = [rhyme]
        if rhyme not in self._vocabulary[gram_struct[-1]]:
            rhyme = self._poet.find_closest(rhyme, self._vocabulary[gram_struct[-1]])
        rec = self._rec_answer((rhyme, gram_struct[-1]), gram_struct[:-1])
        if not rec:
            return None
        answer = answer + rec
        return " ".join(reversed(answer)).capitalize()

    def _rec_answer(self, word, structure):
        if not structure:
            return []
        silly_hash = self._table['hash']
        silly_vector = self._table['vector']
        row = -self._table['table'].getrow(silly_hash[word]).toarray()[0, :]  # negate to use ascending order
        drow = np.arange(len(row))
        stacked = np.array(list(zip(row, drow)), dtype=[('probability', float), ('index', int)])
        row = np.sort(stacked, kind='heapsort', order='probability')
        it = np.nditer(row, flags=['c_index'], op_flags=['readonly', 'readonly'])
        while not it.finished:
            w = silly_vector[it[0]['index']]
            if w[1] == structure[-1] and it[0]['probability']:
                ans = self._rec_answer(w, structure[:-1])
                if ans is not None:
                    return [w[0]] + ans
            it.iternext()
        return None


class GeneralisedRapper(ExhaustiveRapper):
    def __init__(self, grammar, vocabulary, table):
        super().__init__(grammar, vocabulary, table)
        self._poet = Poet(nltki=NltkInterfaceGeneralised)

    def _answer(self, tokens, gram_struct):
        return super()._answer(tokens, gram_struct)
