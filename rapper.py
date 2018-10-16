import numpy as np

import nltk
import re

import requests


class NoRhyme(BaseException):
    pass


class NltkInterface:
    reg_exp = re.compile(r'\w+')

    @classmethod
    def tokenize(cls, sentence):
        tokens = nltk.tokenize.word_tokenize(sentence)
        return list(filter(lambda x: cls.reg_exp.fullmatch(x), tokens))

    @classmethod
    def just_tags(cls, tokens):
        t = nltk.pos_tag(tokens)
        return tuple([x[1] for x in t])

    @classmethod
    def tag_word(cls, word):
        return nltk.pos_tag([word])[0][1]


class Poet:
    uris = ["https://api.datamuse.com/words?rel_rhy={}", "http://rhymebrain.com/talk?function=getRhymes&word={}"]

    @classmethod
    def rhyme(cls, word, pos):
        for uri in cls.uris:
            url = uri.format(word)
            r = requests.get(url)
            json = r.json()
            for j in json:
                word = j['word']
                p = NltkInterface.tag_word(word)
                if p == pos:
                    return word
        raise NoRhyme

    @classmethod
    def rhyme_in(cls, word, pos, bag):
        for uri in cls.uris:
            url = uri.format(word)
            r = requests.get(url)
            json = r.json()
            for j in json:
                word = j['word']
                p = NltkInterface.tag_word(word)
                if p == pos and word in bag:
                    return word
        raise NoRhyme


class MetaRapper(type):
    def __new__(mcs, name, bases, namespace):
        if '_answer' not in namespace:
            raise TypeError('_answer was not implemented in {}'.format(name))
        return super().__new__(mcs, name, bases, namespace)


class BaseRapper(metaclass=MetaRapper):
    """
    Base class to generate a Rapper. Child classes must implement _answer.
    """

    def __init__(self, grammar, predecessors, vocabulary):
        """
        :type grammar: dict
        :type predecessors: dict
        :type vocabulary: dict
        """
        self._grammar = grammar
        self._predecessors = predecessors
        self._vocabulary = vocabulary

    def rap(self, sentence):
        """
        :param sentence: a well formed sentence. Has word characters.
        :type sentence: str
        :return: a rhyming sentence
        :rtype str | None
        """
        tokens = NltkInterface.tokenize(sentence)
        gram_struct = NltkInterface.just_tags(tokens)
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
        super(FastTextRapper, self).__init__(grammar, predecessors, vocabulary)

    def _answer(self, tokens, gram_struct):
        try:
            rhyme = Poet.rhyme(tokens[-1], gram_struct[-1])
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

    def _answer(self, tokens, gram_struct):
        try:
            rhyme = Poet.rhyme_in(tokens[-1], gram_struct[-1], self._vocabulary[gram_struct[-1]])
            answer = [rhyme] + self._rec_answer((rhyme, gram_struct[-1]), gram_struct[:-1])
            return " ".join(reversed(answer)).capitalize()
        except NoRhyme:
            return None

    def _rec_answer(self, word, structure):
        if not structure:
            return []
        silly_hash = self._predecessors['hash']
        silly_vector = self._predecessors['vector']
        row = -self._predecessors['table'].getrow(silly_hash[word]).toarray()[0, :]  # negate to use ascending order
        drow = np.arange(len(row))
        stacked = np.array(list(zip(row, drow)), dtype=[('probability', float), ('index', int)])
        row = np.sort(stacked, kind='heapsort', order='probability')
        row['probability'] *= -1
        it = np.nditer(row, flags=['f_index'])
        while not it.finished:
            w = silly_vector[it[0]['index']]
            if w[1] == structure[-1]:
                ans = self._rec_answer(w, structure[:-1])
                if ans is not None:
                    return [w[0]] + ans
            it.iternext()
        return None
