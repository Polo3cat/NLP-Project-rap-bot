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
        if not json:
            raise NoRhyme
        return


class Rapper:
    def __init__(self, grammar, predecessors, vocabulary, fast_text_keyed_vectors):
        """
        :type grammar: dict
        :type predecessors: dict
        :type vocabulary: dict
        :type fast_text_keyed_vectors: gensim.models.keyedvectors.FastTextKeyedVectors
        """
        self._grammar = grammar
        self._predecessors = predecessors
        self._vocabulary = vocabulary
        self._fast_text_keyed_vectors = fast_text_keyed_vectors

    def rap(self, sentence):
        """
        :param sentence: a well formed sentence. Has word characters.
        :type sentence: str
        :return: a rhyming sentence
        :rtype str | None
        """
        tokens = NltkInterface.tokenize(sentence)
        gram_struct = NltkInterface.just_tags(tokens)
        try:
            rhyme = Poet.rhyme(tokens[-1], gram_struct[-1])
        except NoRhyme:
            return None
        answer = [rhyme]
        previous_w = rhyme
        for tag in reversed(gram_struct[:-1]):
            if previous_w not in self._predecessors or \
                    tag not in self._predecessors[previous_w]:
                previous_w = self._find_similar(previous_w, tag)
            else:
                previous_w = self._predecessors[previous_w][tag]
            answer.append(previous_w)
        return " ".join(reversed(answer)).capitalize()

    def _find_similar(self, w0, desired_pos):
        w1 = None
        best_similarity = 10.
        for w in self._vocabulary[desired_pos]:
            similarity = self._fast_text_keyed_vectors.similarity(w0, w)
            if best_similarity < similarity:
                best_similarity = similarity
                w1 = w
        return w1
