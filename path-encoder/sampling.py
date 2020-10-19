import sys
import random
import numpy as np
from tqdm import tqdm

in_fname = sys.argv[1]
out_fname = sys.argv[2]
method = sys.argv[3]
num_neg = int(sys.argv[4])

out_fname += '.' + method
if method not in ['r', 'c', 'n']:
    exit(0)
elif method == 'r':

    raw_instances = []
    with open(in_fname) as in_f:
        for line in in_f:
            raw_instances.append(line.strip().split('\t'))
    raw_instances = raw_instances
    instances = []
    for idx, raw_inst in enumerate(raw_instances):
        instances.append(raw_inst + ['1'])
        print(instances[-1])

        rand_idxes = []
        for i in range(num_neg):
            rand_idx = random.randint(0, len(raw_instances) - 1)
            while rand_idx == idx or rand_idx in rand_idxes:
                rand_idx = random.randint(0, len(raw_instances) - 1)
            rand_idxes.append(rand_idx)

        try:
            for idx in rand_idxes:
                instances.append(raw_instances[idx][0:-1] + [raw_inst[-1]] + ['0'])
        except IndexError:
            print(rand_idx, len(raw_instances))
            raise IndexError

    with open(out_fname, 'w') as out_f:
        for inst in instances:
            for item in inst:
                out_f.write(item)
                out_f.write('\t')
            out_f.write('\n')

elif method == 'n':
    synset2embedding = {}

    with open('ft-embs-all-lower.vec') as in_f:
        lines = tqdm(in_f.readlines(), desc=f'loading embeddings from ft-embs-all-lower.vec', ascii=True)
        for line in lines:
            l = line.strip().split()
            if len(l) > 1:
                synset2embedding[l[0]] = np.asarray([float(item) for item in l[1:]])
        lines.set_description('loading embeddings ---done!')

    embeddings = []
    index2synset = []
    all_synset_path = []
    uni_synset_path = []
    with open(in_fname) as in_f:
        lines = tqdm(in_f.readlines(), desc=f'loading instances from {in_fname}', ascii=True)
        for line in lines:
            l = line.strip().split()
            if len(l) > 1:
                path = l[:-2]
                hyper = l[-2]
                target = l[-1]
                all_synset_path.append((path, hyper, target))
                path_hyper = '\t'.join(l[:-1])
                if path_hyper not in uni_synset_path:
                    uni_synset_path.append(path_hyper)
                    if hyper not in index2synset and hyper in synset2embedding:
                        index2synset.append((path, hyper))
                        embeddings.append(synset2embedding[hyper])
    embeddings = np.asarray(embeddings)

    inst_out = []
    print(len(all_synset_path), len(uni_synset_path), len(index2synset))

    all_synset_path_t = tqdm(all_synset_path, ascii=True)
    for idx, inst in enumerate(all_synset_path_t):
        path, hyper, target = inst
        inst_out.append((path, hyper, target, '1'))
        if hyper not in synset2embedding:
            rand_idxes = []
            for i in range(num_neg):
                rand_idx = random.randint(0, len(uni_synset_path) - 1)
                while rand_idx == idx or rand_idx in rand_idxes:
                    rand_idx = random.randint(0, len(uni_synset_path) - 1)
                rand_idxes.append(rand_idx)
                inst_out.append((uni_synset_path[rand_idx][0], uni_synset_path[rand_idx][1], target, '0'))
        else:
            # print(hyper)
            hyper_embedding = synset2embedding[hyper]
            hyper_embedding = np.tile(hyper_embedding, [len(index2synset), 1])
            diff = hyper_embedding - embeddings
            dist = np.linalg.norm(diff, axis=1).squeeze()
            topk = list(np.argsort(dist)[:num_neg+1])[1:]
            # print(topk)
            topk_hyper = []
            for t in topk:
                inst_out.append((index2synset[t][0], index2synset[t][1], target, '0'))
                topk_hyper.append(index2synset[t][1])
            # print(target, hyper, topk_hyper)

    with open(out_fname, 'w') as out_f:
        for path, hyper, target, label in inst_out:
            for item in path:
                out_f.write(item)
                out_f.write('\t')
            out_f.write(hyper)
            out_f.write('\t')
            out_f.write(target)
            out_f.write('\t')
            out_f.write(label)
            out_f.write('\t')
            out_f.write('\n')

elif method == 'c':
    all_path_target = {}
    all_path = []
    all_target = []
    all_target_path = {}
    with open(in_fname) as in_f:
        lines = tqdm(in_f.readlines(), desc=f'loading instances from {in_fname}', ascii=True)
        for line in lines:
            l = line.strip().split()
            if len(l) > 1:
                target = l[-1]
                path_hyper = '\t'.join(l[:-1])
                if path_hyper in all_path_target:
                    if target not in all_path_target[path_hyper]:
                        all_path_target[path_hyper].append(target)
                else:
                    all_path_target[path_hyper] = [target]
                    all_path.append(path_hyper)
                if target in all_target_path:
                    if path_hyper not in all_target_path:
                        all_target_path[target].append(path_hyper)
                else:
                    all_target_path[target] = [path_hyper]
                    all_target.append(target)

    num_neg_half = num_neg // 2

    inst_out = []
    for path in all_path:
        for target in all_path_target[path]:
            inst_out.append((path, target, '1'))
            rand_idxes_target = []
            for i in range(num_neg_half):
                rand_idx = random.randint(0, len(all_target) - 1)
                while all_target[rand_idx] in all_path_target[path] or rand_idx in rand_idxes_target:
                    rand_idx = random.randint(0, len(all_target) - 1)
                rand_idxes_target.append(rand_idx)
            rand_idxes_path = []
            for i in range(num_neg_half):
                rand_idx = random.randint(0, len(all_path) - 1)
                while all_path[rand_idx] in all_target_path[target] or rand_idx in rand_idxes_path:
                    rand_idx = random.randint(0, len(all_path) - 1)
                rand_idxes_path.append(rand_idx)
            for inst in rand_idxes_target:
                inst_out.append((path, all_target[inst], '0'))
            for inst in rand_idxes_path:
                inst_out.append((all_path[inst], target, '0'))

    with open(out_fname, 'w') as out_f:
        for path, target, label in inst_out:
            out_f.write(path)
            out_f.write('\t')
            out_f.write(target)
            out_f.write('\t')
            out_f.write(label)
            out_f.write('\n')
