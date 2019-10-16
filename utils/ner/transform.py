import sys
from tqdm import tqdm

if len(sys.argv) < 3:
    print("provide input and output file name")
    sys.exit()
input_file, output_file = sys.argv[1:3]

rep = {'O': '0', 'LOCATION': '1', 'PERSON': '2', 'ORGANIZATION': '3'}

num_lines = sum([1 for line in open(input_file)])
with open(input_file) as f, open(output_file, 'w') as of:
    for line in tqdm(f, total=num_lines):
        tags = line.strip().split(' ')

        of.write(' '.join([rep[x] for x in tags]))
        of.write('\n')
