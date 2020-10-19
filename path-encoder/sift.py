lines = []
with open('data/split/splited/valid_verb_path') as in_f:
    for line in in_f:
        lines.append(line.strip().split('\t'))

wn18 = {}
with open('data/split/raw/valid_verb') as in_f:
    for line in in_f:
        l = line.strip().split('\t')
        if l[1] != '_hypernym':
            continue
        if l[0] not in wn18:
            wn18[l[0]] = [l[2]]
        else:
            wn18[l[0]].append(l[2])

print(wn18)
with open('data/split/splited/valid_verb_path_18rr', 'w', encoding='utf-8') as out_f:
    for line in lines:
        if line[-2] not in wn18[line[-1]]:
            pass
        else:
            for l in line:
                out_f.write(l)
                out_f.write('\t')
            out_f.write('\n')
