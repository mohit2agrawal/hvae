import csv

### first 1000 0 sentences
### then  1000 1 sentences

data = open('generated_negative.txt').readlines()
data.extend(open('generated_positive.txt').readlines())

output_fn = 'stanford_generated_mod.tsv'

with open(output_fn, 'wb') as tsvfile:
    tsvw = csv.writer(tsvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)

    tsvw.writerow(['index', 'sentence'])

    for i, sent in enumerate(data):
        tsvw.writerow([i, sent.strip()])
