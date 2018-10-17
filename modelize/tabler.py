#!/bin/python3

from scipy.sparse import lil_matrix


class Tabler:
    def __init__(self, counter):
        self._types_count = counter.get_types_counter()
        self._bigrams_count = counter.get_bigrams_counter()
        self._total_w = counter.get_total_w()

    def result(self):
        sh_count = 0
        silly_hash = {}
        silly_vector = [None]*len(self._types_count)
        prob_table = lil_matrix((len(self._types_count), len(self._types_count)))

        # P(w1|w0) = C(w0 w1) / C(w0)
        for b in self._bigrams_count:
            w0 = b[0]
            w1 = b[1]
            if w0 not in silly_hash:
                silly_hash[w0] = sh_count
                silly_vector[sh_count] = w0
                sh_count += 1
            if w1 not in silly_hash:
                silly_hash[w1] = sh_count
                silly_vector[sh_count] = w1
                sh_count += 1
            row = silly_hash[w0]
            col = silly_hash[w1]
            prob_table[row, col] = (self._bigrams_count[b] + 1) / (self._types_count[w0] + self._total_w)

        prob_table = prob_table.tocsr(copy=True)

        for w in self._types_count:
            if w not in silly_hash:
                silly_hash[w] = sh_count
                silly_vector[sh_count] = w
                sh_count += 1
        return {'table': prob_table, 'hash': silly_hash, 'vector': silly_vector}
