#! /bin/python3

import csv
import sys


source = sys.argv[1]
output = sys.argv[2]
genre  = sys.argv[3]
amount = int(sys.argv[4])
with open(source, newline='') as source_csv, open(output, newline='', mode='w') as output_csv:
    csvreader = csv.DictReader(source_csv, dialect='unix')
    csvwriter = csv.DictWriter(output_csv, dialect='unix', fieldnames=csvreader.fieldnames)
    csvwriter.writeheader()
    count = 0
    for row in csvreader:
        if row['genre'] == genre:
            csvwriter.writerow(row)
            count += 1
        if count == amount:
            break
