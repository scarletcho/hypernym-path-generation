import sys

in_fname = sys.argv[1]
out_fname = sys.argv[2]
HIT = int(sys.argv[3])
dataset = sys.argv[4]

synset2labels = {}
ixxx = []
with open(f'data/{dataset}-hp-all.txt') as in_f:
    for line in in_f:
        l = line.strip().split('\t')
        synset2labels[l[0]] = l[1:]
        if l[0] not in ixxx:
            ixxx.append(l[0])
        else:
            # print(f'Warning \'\'{l[0]}\'\' has more than one direct hypernym')
            pass

pred = {}
with open(in_fname) as in_f:
    for idx, line in enumerate(in_f):
        hit = idx % HIT
        if hit not in pred:
            pred[hit] = []
        pred[hit].append(line.strip().split())
# print(pred.keys())

printed = [[] for i in range(HIT)]

_line_ = {}
# print(HIT)
for h in range(HIT):
    # print(h)
    if h not in _line_:
        _line_[h] = []
    for i in range(len(pred[HIT - 1])):
        # print(i)
        if pred[h][i][-1] not in printed[h]:
            if len(pred[h][i]) != 0:

                printed[h].append(pred[h][i][-1])
                _line_[h].append((pred[h][i][-1], synset2labels[pred[h][i][-1]], pred[h][i][-2]))
# print(len(_line_))
# print(_line_)
with open(out_fname + '.format', 'w') as out_f:
    for item in _line_[0]:
        node, label, pred = item
        # print(node, label, pred)
        out_f.write(node)
        out_f.write('\t')
        for l in label:
            out_f.write(l)
            out_f.write('\t')
        out_f.write(pred)
        out_f.write('\n')

# print(len(_line_[1]))
if HIT > 1:
    with open(f'{out_fname}_@hit{HIT}.format' , 'w') as out_f:
        for syn in range(len(_line_[0])):
            for i in range(HIT):
                try:
                    node, label, pred = _line_[i][syn]
                except IndexError:
                    print(_line_.keys())
                    print(syn, i, _line_[i])
                    exit()
                # print(node, label, pred)
                out_f.write(node)
                out_f.write('\t')
                for l in label:
                    out_f.write(l)
                    out_f.write('\t')
                out_f.write(pred)
                out_f.write('\n')

