#!/bin/python3

import nltk
import sys
import csv
import pickle
import re


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


class Grammer:
    def __init__(self):
        self._graph = WDG()

    def feed(self, lines):
        previous = lines[0]
        for line in lines[1:]:
            self._graph.add_edge(previous, line)
            previous = line

    def result(self):
        ret = {}
        for node, edges in self._graph.nodes.items():
            ret[node] = max(edges)
        return ret
