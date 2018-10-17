#!/bin/python3


class Predecessorer:
    def __init__(self, counter):
        self._types_count = counter.get_types_counter()
        self._bigrams_count = counter.get_bigrams_counter()

    def result(self):
        conditional_probabilities = {}
        type_pos_map = {}

        # P(w1|w0) = C(w0 w1) / C(w0)
        for b in self._bigrams_count:
            conditional_probabilities[b] = self._bigrams_count[b] / self._types_count[b[0]]

        # For each w0 keep the highest P for each PoS tag
        for bigram, probability in conditional_probabilities.items():
            w0 = bigram[0]
            w1_type = bigram[1][0]
            w1_pos = bigram[1][1]
            if w0 not in type_pos_map:
                type_pos_map[w0] = {w1_pos: (w1_type, probability)}
            elif w1_pos not in type_pos_map[w0] or type_pos_map[w0][w1_pos][1] > probability:
                type_pos_map[w0][w1_pos] = (w1_type, probability)

        # strip away probabilities as they won't be used anymore
        w0_pos_to_pos_w1 = {}
        for w0, words1 in type_pos_map.items():
            aux_dict = {}
            for k, v in type_pos_map[w0].items():
                aux_dict[k] = v[0]
            w0_pos_to_pos_w1[w0] = aux_dict
