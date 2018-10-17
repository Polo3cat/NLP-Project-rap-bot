import argparse
import pickle
from rapper import FastTextRapper

parser = argparse.ArgumentParser(prog="The Rapper")
parser.add_argument('-m', help="Model", required=True)
parser.add_argument('-f', help="Fast-text vectors", required=True)

args = parser.parse_args()

with open(args.m, mode='br') as f:
    model = pickle.load(f)
with open(args.f, mode='br') as f:
    f_t_vectors = pickle.load(f)

grammar = model['grammar']
predecessors = model['predecessors']
vocabulary = model['vocabulary']

rapper = FastTextRapper(grammar, predecessors, vocabulary, f_t_vectors)
while True:
    try:
        s = input("->")
    except EOFError:
        break
    if s and not s.isspace():
        a = rapper.rap(s)
        print(a) if a else print("(Info.) No rhyme found")
print('\nBye!')

