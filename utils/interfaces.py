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


class NltkInterfaceGeneralised(NltkInterface):
    reg_exp = re.compile(r'\w+')

    general ={
        '$': '$',
        "''": "''",
        "``":"``",
        '(': '(',
        ')': ')',
        ',': ',',
        '--': '--',
        '.': '.',
        ':': ':',
        'CC': 'CC',
        'CD': 'CD',
        'EX': 'EX',
        'FW': 'FW',
        'IN': 'IN',
        'DT': 'DT',
        'LS': 'LS',
        'MD': 'MD',
        'PDT': 'PDT',
        'RP': 'RP',
        'SYM': 'SYM',
        'TO': 'TO',
        'UH': 'UH',
        'JJ': 'ADJ',
        'JJR': 'ADJ',
        'JJS': 'ADJ',
        'NN': 'NN',
        'NNP': 'NN',
        'NNPS': 'NN',
        'NNS': 'NN',
        'POS': 'NN',
        'PRP': 'PRP',
        'PRP$': 'PRP',
        'RB': 'RB',
        'RBR': 'RB',
        'RBS': 'RB',
        'VB': 'VB',
        'VBD': 'VB',
        'VBG': 'VB',
        'VBN': 'VB',
        'VBP': 'VB',
        'VBZ': 'VB',
        'WDT': 'WH',
        'WP': 'WH',
        'WP$': 'WH',
        'WRB': 'WH'
    }

    @classmethod
    def tokenize(cls, sentence):
        tokens = nltk.tokenize.word_tokenize(sentence)
        return list(filter(lambda x: cls.reg_exp.fullmatch(x), tokens))

    @classmethod
    def tag(cls, tokens):
        return [(x[0], cls.general[x[1]]) for x in nltk.pos_tag(tokens)]

    @classmethod
    def strip_words(cls, tagged_tokens):
        return tuple([x[1] for x in tagged_tokens])

    @classmethod
    def just_tags(cls, tokens):
        t = nltk.pos_tag(tokens)
        return tuple([cls.general[x[1]] for x in t])

    @classmethod
    def tag_word(cls, word):
        return cls.general[nltk.pos_tag([word])[0][1]]
