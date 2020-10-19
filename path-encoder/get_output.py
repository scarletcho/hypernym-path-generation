import sys
import os
import subprocess
from tqdm import tqdm

arguments = len(sys.argv) - 1

path = sys.argv[1]
HIT = sys.argv[2]
dataset = 'valid'
if arguments > 2:
    dataset = 'test' if sys.argv[3].lower() == 'test' else 'valid'


def write_to_file(out_file, metric, data):
    out_file.write(metric)
    out_file.write('\t')
    out_file.write(str(data[0]))
    out_file.write('\t')
    out_file.write(data[1])
    out_file.write('\t')
    out_file.write(str(data[2:]))
    out_file.write('\n')


if not os.path.isdir(path):
    exit(1000)
out_dirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
# print(out_dirs)
for folder in out_dirs:
    f_name_wo_path = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith('.out')]
    out_files = [os.path.join(folder, f) for f in f_name_wo_path]
    if len(out_files) == 0:
        continue
    # print(out_files)

    processes = []
    for file in out_files:
        p = subprocess.Popen(["python", "format.py", file, file[:-4], HIT, dataset])
        processes.append(p)

    for process in processes:
        _, __ = process.communicate()

    processes = []
    for file in out_files:
        # subprocess.run(["python", "process_out.py", file, HIT])
        p = subprocess.Popen(["python", "process_out.py", file, HIT])
        processes.append(p)
        # _, __ = p.communicate()
    processes_t = tqdm(processes, ascii=True)
    for process in processes_t:
        result, err = process.communicate()
        # process.wait()

    processes = []
    for idx, file in enumerate(out_files):
        out_fname = file[:-4]
        p3 = subprocess.Popen(["python", "score.py", folder, out_fname + '.result', out_fname + '.score'], stdout=subprocess.PIPE)
        processes.append((f_name_wo_path[idx][:-4], p3))


    f_names = []
    scores = []
    for out_fname, process in processes:
        result, err = process.communicate()
        result = result.decode("utf-8").strip().split()
        f_names.append(out_fname)
        scores.append([float(i) for i in result])

    file_score = list(zip(f_names, scores))
    rank = []
    for metric in range(6):
        rank.append(sorted(file_score, key=lambda x: x[1][metric])[-1])

    with open(folder + '.summary', 'w') as out_f:
        out_f.write("epoch\th@1_dev\th@1_wn\tlex_dev\tlex_wn\twup_dev\twup_wn_max\n")
        for i in range(6):
            out_f.write(f'{rank[i][0]}\t')
            out_f.write('\t'.join([str(s) for s in rank[i][1]]))
            out_f.write('\n')


'''
        scores = (result[1], result[10], result[13], result[16])

        lines = []
        synsets = []
        predicted = []
        targets = []
        hit = int(HIT)

        with open(f'{out_fname}_@hit{HIT}.format') as in_f:
            for idx, line in enumerate(in_f):
                l = line.strip().split()
                if idx % hit == 0:
                    predicted.append([l[-1]])
                    synsets.append(l[0])
                    targets.append(l[-2])
                else:
                    predicted[-1].append(l[-1])

        correct = 0
        for i in range(len(targets)):
            outputs = predicted[i]
            target = targets[i]
            if target in outputs:
                # print(outputs, target)
                correct += 1

    scores = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith('.score')]
    print(scores)
    best_h1_wn18rr = None
    best_h10_wn18rr = None
    best_h1_wordnet = None
    best_h1_lex = None
    best_h1_wupalmer = None
    for score in scores:
        with open(os.path.join(folder, score)) as score_f:
            lines = [float(line.strip()) for line in score_f.readlines()[1::2]]
            print(lines)
            if best_h1_wn18rr is None:
                best_h1_wn18rr = [lines[0], score[:-6], lines[0], lines[1], lines[2], lines[3], lines[4]]
                best_h1_wordnet = [lines[1], score[:-6], lines[0], lines[1], lines[2], lines[3], lines[4]]
                best_h1_lex = [lines[2], score[:-6], lines[0], lines[1], lines[2], lines[3], lines[4]]
                best_h1_wupalmer = [lines[3], score[:-6], lines[0], lines[1], lines[2], lines[3], lines[4]]
                best_h10_wn18rr = [lines[4], score[:-6], lines[0], lines[1], lines[2], lines[3], lines[4]]
            else:
                if lines[0] > best_h1_wn18rr[0]:
                    best_h1_wn18rr = [lines[0], score[:-6], lines[0], lines[1], lines[2], lines[3], lines[4]]
                if lines[1] > best_h1_wordnet[0]:
                    best_h1_wordnet = [lines[1], score[:-6], lines[0], lines[1], lines[2], lines[3], lines[4]]
                if lines[2] > best_h1_lex[0]:
                    best_h1_lex = [lines[2], score[:-6], lines[0], lines[1], lines[2], lines[3], lines[4]]
                if lines[3] > best_h1_wupalmer[0]:
                    best_h1_wupalmer = [lines[3], score[:-6], lines[0], lines[1], lines[2], lines[3], lines[4]]
                if lines[4] > best_h10_wn18rr[0]:
                    best_h10_wn18rr = [lines[4], score[:-6], lines[0], lines[1], lines[2], lines[3], lines[4]]

    score_by_epoch = [best_h1_wn18rr, best_h1_wordnet, best_h1_lex, best_h1_wupalmer, best_h10_wn18rr]
    with open(folder + '.summary', 'w') as out_f:
        for i, met in enumerate(('best_h1_wn18rr', 'best_h1_wordnet', 'best_h1_lex', 'best_h1_wupalmer',
                                 'best_h10_wn18rr')):
            write_to_file(out_f, met, score_by_epoch[i])
'''