import re

import nltk


class NltkInterface:
    reg_exp = re.compile(r'\w+')

    @classmethod
    def tokenize(cls, sentence):
        tokens = nltk.tokenize.word_tokenize(sentence)
        return list(filter(lambda x: cls.reg_exp.fullmatch(x), tokens))

    @classmethod
    def tag(cls, tokens):
        return nltk.pos_tag(tokens)

    @classmethod
    def strip_words(cls, tagged_tokens):
        return tuple([x[1] for x in tagged_tokens])

    @classmethod
    def just_tags(cls, tokens):
        t = nltk.pos_tag(tokens)
        return tuple([x[1] for x in t])

    @classmethod
    def tag_word(cls, word):
        return nltk.pos_tag([word])[0][1]