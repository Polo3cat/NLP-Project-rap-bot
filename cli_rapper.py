import argparse
import pickle
from rapper import FastTextRapper

parser = argparse.ArgumentParser(prog="The Rapper")
parser.add_argument('-g', help="Grammar", required=True)
parser.add_argument('-p', help="Predecessors", required=True)
parser.add_argument('-v', help="Vocabulary", required=True)
parser.add_argument('-f', help="Fast-text vectors", required=True)

args = parser.parse_args()

with open(args.g, mode='br') as f:
    grammar = pickle.load(f)
with open(args.p, mode='br') as f:
    predecessors = pickle.load(f)
with open(args.v, mode='br') as f:
    vocabulary = pickle.load(f)
with open(args.f, mode='br') as f:
    f_t_vectors = pickle.load(f)

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

