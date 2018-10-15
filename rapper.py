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
    uri = "https://api.datamuse.com/words?rel_rhy={}"

    @classmethod
    def rhyme(cls, word, pos):
        url = cls.uri.format(word)
        r = requests.get(url)
        json = r.json()
        for j in json:
            word = j['word']
            p = NltkInterface.tag_word(word)
            if p == pos:
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
    def _answer(self, tokens, gram_struct):
        pass
