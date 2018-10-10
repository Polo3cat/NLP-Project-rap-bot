#! /bin/python3

import pickle
import sys
import nltk
import Levenshtein
import requests
import re


class NoRhyme(BaseException):
    pass


class Empty(BaseException):
    pass


def find_similar_with_tag(word, w_pos, obj_pos, posteriori_map, candidates):
    if not candidates:
        raise Empty
    for w in candidates:
        if w != word:
            ret = w
            break
    best = Levenshtein.distance(word, ret)
    for w in candidates:
        if (w, w_pos) in posteriori_map and obj_pos in posteriori_map[(w, w_pos)]:
            d = Levenshtein.distance(word, w)
            if d < best:
                best = d
                ret = w
    return ret


def fetch_rhyme(token, pos):
    url = "https://api.datamuse.com/words?rel_rhy={}".format(token)
    r = requests.get(url)
    json = r.json()
    for j in json:
        p = nltk.pos_tag([j['word']])
        if p[0][1] == pos:
            return p[0]
    if not json:
        raise NoRhyme
    return nltk.pos_tag([json[0]['word']])[0]


def leven_distance(str1, str2):
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


def find_similar_grammar(g0, bag):
    best = 10000
    ret = None
    for g in bag:
        d = leven_distance(g0, g)
        if d < best:
            best = d
            ret = g
    return ret


grammar_file = sys.argv[1]
vocabulary_file = sys.argv[2]

with open(grammar_file, mode='br') as f:
    grammar = pickle.load(f)

with open(vocabulary_file, mode='br') as f:
    vocabulary = pickle.load(f)

types_by_pos = vocabulary['pos']
mapping = vocabulary['mapping']

print('Welcome to the interactive rapper. Press ctrl+D to exit')
while True:
    try:
        sample = input('Your sentence: ')
        tokens = nltk.tokenize.word_tokenize(sample)
        tokens = list(filter(lambda x: re.fullmatch(r'\w+', x), tokens))
        ttags = nltk.pos_tag(tokens)
        tags = tuple([x[1] for x in ttags])
        if not sample or not ttags or not tags:
            print('(Info.) You forgot to write something!')
            continue
        if tags not in grammar:
            answer_grammar = find_similar_grammar(tags, grammar.keys())
        else:
            answer_grammar = grammar[tags]
        try:
            rhyme = fetch_rhyme(ttags[-1][0], answer_grammar[-1])
        except NoRhyme as e:
            print("(Info.) Couldn't find a rhyme for that!")
            continue
        answer = [rhyme[0]]
        previous = rhyme[0]
        prev_pos = rhyme[1]
        try:
            for tag in reversed(answer_grammar[:-1]):
                if previous not in mapping or tag not in mapping[previous]:
                    previous = find_similar_with_tag(previous, prev_pos, tag, mapping, types_by_pos[tag])
                else:
                    previous = mapping[previous][tag]
                answer += [previous]
                prev_pos = tag
        except Empty as e:
            print("(Warning!) Model too small!")
        print(" ".join(reversed(answer)))
    except EOFError as e:
        print()
        print('Bye!')
        break
