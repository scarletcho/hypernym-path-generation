if __name__ == '__main__':
    synset_to_path = {}
    with open('all_paths') as in_f:
        for line in in_f:
            l = line.strip().split('\t')
            # print(l[-1])
            if l[-1] not in synset_to_path:
                synset_to_path[l[-1]] = [l[0:-1]]
            else:
                synset_to_path[l[-1]].append(l[0:-1])
    words = []
    words_unknown = []
    with open('test-hp-all.txt') as in_f:
        for line in in_f:
            l = line.strip().split('\t')
            # print(l[0])
            if l[0] in synset_to_path:
                words.append((l[0], synset_to_path[l[0]]))
            else:
                words_unknown.append(words_unknown)

    with open('test_full_path', 'w') as out_f:
        for word, paths in words:
            for idx, path in enumerate(paths):
                for node in path:
                    out_f.write(node)
                    out_f.write('\t')
                out_f.write(word)
                if idx < len(paths) - 1:
                    out_f.write('\n')
            out_f.write('\n')

    print(words_unknown)
    print(synset_to_path['land_reform.n.01'])