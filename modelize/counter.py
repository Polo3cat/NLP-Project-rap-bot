class Counter:
    def __init__(self):
        self._types_counter = {}
        self._vocabulary = {}
        self._bigrams_counter = {}
        self._total_w = 0

    def vocabulary(self):
        return self._vocabulary

    def feed(self, t_tokens):
        t_tokens.reverse()
        for t in t_tokens:
            self._types_counter[t] = 1 + self._types_counter.get(t, 0)
            self._vocabulary[t[1]] = self._vocabulary.get(t[1], set()) | {t[0]}
            self._total_w += 1
        for b in zip(t_tokens, t_tokens[1:]):
            self._bigrams_counter[b] = 1 + self._bigrams_counter.get(b, 0)

    def get_types_counter(self):
        return self._types_counter

    def get_bigrams_counter(self):
        return self._bigrams_counter

    def get_total_w(self):
        return self._total_w
