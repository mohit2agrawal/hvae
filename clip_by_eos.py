import sys

if len(sys.argv) < 3:
    print("provide input and output file name")
    sys.exit()
input_file, output_file = sys.argv[1:3]

with open(input_file) as inp_f, open(output_file, 'w') as out_f:
    tag = '<EOS>'
    try:
        l = inp_f.readline()
        _ = list(map(int, l.strip().split(' ')))
        tag = '5'
    except:
        pass

    inp_f.seek(0)
    for line in inp_f:
        out_f.write(line[:line.find(tag)])
        out_f.write('\n')
