#!/bin/python3

import nltk
import sys
import csv
import pickle


class WDG:
    """
    Weighted Directed Graph. Very simple implementation using dictionaries.
    """
    def __init__(self):
        self.nodes = {}

    def __repr__(self):
        return "{!r}".format(self.nodes)

    def add_node(self, value):
        if value not in self.nodes:
            self.nodes[value] = {}

    def add_edge(self, n0, n1):
        if n0 in self.nodes:
            if n1 in self.nodes[n0]:
                self.nodes[n0][n1] += 1
            else:
                self.nodes[n0][n1] = 1
        else:
            self.nodes[n0] = {n1: 1}


def lines_to_graph(lines, graph):
    # take care of the first line
    tokens = nltk.tokenize.word_tokenize(lines[0])
    previous = tuple([x[1] for x in nltk.pos_tag(tokens)])
    for verse in lines[1:]:
        tokens = nltk.tokenize.word_tokenize(verse)
        tags = tuple([x[1] for x in nltk.pos_tag(tokens)])
        graph.add_edge(previous, tags)
        previous = tags


source = sys.argv[1]
savefile = sys.argv[2]
corpus_tags = WDG()
with open(source, newline='') as sourcefile:
    csvreader = csv.DictReader(sourcefile, dialect='unix')
    for row in csvreader:
        lyrics = row['lyrics']
        if not lyrics:
            continue
        lines_to_graph(lyrics.splitlines(), corpus_tags)

# keep edges in graph with highest weight
skewed_tags = {}
for node in iter(corpus_tags.nodes):
    try:
        skewed_tags[node] = max(corpus_tags.nodes[node])
    except ValueError as e:
        print(node, 'has no verses following it.')

# save created pairing of grammatical structures in a file so we can retrieve it later
with open(savefile, mode='bw') as sf:
    pickle.dump(skewed_tags, sf)

print(skewed_tags)