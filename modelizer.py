#! /bin/python3

import argparse
import pickle
import multiprocessing
from csv import DictReader

from modelize.counter import Counter
from modelize.grammer import Grammer
from modelize.vectorizer import Vectorizer
from modelize.predecessorer import Predecessorer
from modelize.tabler import Tabler
import utils.interfaces


parser = argparse.ArgumentParser(prog="Modelizer")
parser.add_argument('-o', help="output file for model", required=True)
parser.add_argument('-i', help="input csv file", required=True)
parser.add_argument('-m', help="model to generate: table or vector", default='table', choices=['table', 'vector'])
parser.add_argument('-g', help="use a more general grammar", action='store_true')

args = parser.parse_args()

nltki = utils.interfaces.NltkInterface
if args.g:
    nltki = utils.interfaces.NltkInterfaceGeneralised

cpus = multiprocessing.cpu_count()
ls = 0
with open(args.i, newline='') as i:
    csvreader = DictReader(i, dialect='unix')
    for l in csvreader:
        ls += 1

ch_size = ls / cpus
remainder = ls % cpus
slices = []
for i in range(cpus):
    begin = (i * ch_size) + (i if i < remainder else remainder)
    end = begin+ch_size+(i < remainder)
    slices.append([int(begin), int(end)])


def worker(init, fin, input_name, output_name, m):
    counter = Counter()
    grammer = Grammer()

    with open(input_name, newline='') as input_file:
        csvreader = DictReader(input_file, dialect='unix')
        count = 0
        song = next(csvreader)
        while count < init:
            count += 1
            song = next(csvreader)
        while count < fin:
            print(multiprocessing.current_process().name, 'Doing:', song['index'])
            lyrics = song['lyrics']
            if not lyrics:
                try:
                    song = next(csvreader)
                    count += 1
                    continue
                except StopIteration:
                    break;
            lines = lyrics.splitlines()
            song_tags = []
            for line in lines:
                tokens = nltki.tokenize(line)
                tagged_tokens = nltki.tag(tokens)
                song_tags.append(nltki.strip_words(tagged_tokens))
                counter.feed(tagged_tokens)
            grammer.feed(song_tags)
            count += 1
            try:
                song = next(csvreader)
            except StopIteration:
                break

    output = {'grammar': grammer.result(), 'vocabulary': counter.vocabulary(), }

    if m == 'vector':
        predecessorer = Predecessorer(counter)
        output['predecessors'] = predecessorer.result()
        with open(input_name, newline='') as input_file:
            csvreader = DictReader(input_file, dialect='unix')
            vectorizer = Vectorizer(csvreader)
            output['ftvectors'] = vectorizer.result()
    else:
        tabler = Tabler(counter)
        output['table'] = tabler.result()

    with open(output_name, mode='bw') as f:
        pickle.dump(output, f)


processes = []
file_names = []
num = 0
for slice_ in slices:
    fn = args.o + str(num)
    p = multiprocessing.Process(target=worker, args=(*slice_, args.i, fn, args.m))
    processes.append(p)
    file_names.append(fn)
    num += 1
for p in processes:
    p.start()
for p in processes:
    p.join()

for fn in file_names:
    
