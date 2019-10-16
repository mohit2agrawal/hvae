'''verify if the length of list after splitting by space
is equal for sentences and labels
'''

# inp_a = 'a.txt'
# inp_b = 'aa.txt'
inp_a = 'data.txt'
inp_b = 'labels.txt'

num_lines_a = sum([1 for line in open(inp_a)])
num_lines_b = sum([1 for line in open(inp_b)])
print(num_lines_a, num_lines_b)

with open(inp_a) as af, open(inp_b) as bf:
    i = 0
    while i < num_lines_a:
        a = af.readline()
        b = bf.readline()
        al = len(a.strip().split(' '))
        bl = len(b.strip().split(' '))
        if al != bl:
            print(i)
            print(al, bl)
            print(a)
            print(b)
            break
        i += 1
