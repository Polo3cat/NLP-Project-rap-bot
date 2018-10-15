#! /bin/python3

import csv
import sys


source = sys.argv[1]
output = sys.argv[2]
genre  = sys.argv[3]
amount = int(sys.argv[4])
with open(source, newline='') as source_csv, open(output, newline='', mode='w') as output_f:
    csvreader = csv.DictReader(source_csv, dialect='unix')
    count = 0
    for row in csvreader:
        if row['genre'] == genre:
            output_f.write(row['lyrics'])
            count += 1
        if count == amount:
            break
