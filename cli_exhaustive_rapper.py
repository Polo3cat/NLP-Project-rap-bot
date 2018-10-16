import argparse
import pickle
import numpy as np
from rapper import ExhaustiveRapper

parser = argparse.ArgumentParser(prog="The Rapper")
parser.add_argument('-g', help="Grammar", required=True)
parser.add_argument('-t', help="Table", required=True)
parser.add_argument('-v', help="Vocabulary", required=True)
parser.add_argument('-p', help="Hash", required=True)

args = parser.parse_args()

with open(args.g, mode='br') as f:
    grammar = pickle.load(f)
with open(args.t, mode='br') as f:
    table = np.load(f)
with open(args.v, mode='br') as f:
    vocabulary = pickle.load(f)
with open(args.p, mode='br') as f:
    hash_vector = pickle.load(f)

predecessors = {'table': table, **hash_vector}
rapper = ExhaustiveRapper(grammar, predecessors, vocabulary)
while True:
    try:
        s = input("->")
    except EOFError:
        break
    if s and not s.isspace():
        a = rapper.rap(s)
        print(a) if a else print("(Info.) No rhyme found")
print('\nBye!')