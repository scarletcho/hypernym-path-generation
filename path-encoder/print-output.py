# rel	node(hypo)	node_lexname	pred(hyper)_str	pred_lexname	gold_str	gold_lex	is_gold_dev	is_gold_wn	lex_ident	wup_similarity
from itertools import islice
from nltk.corpus import wordnet as wn
import re
import sys, time

in_fname = sys.argv[1]
out_fname = sys.argv[2]

# in_fname = 'node-rel-gold-pred.txt'
# out_fname = 'output.txt'

with open(in_fname , 'r') as f:
    corpus = []
    for line in f:
        line = re.sub('\n', '', line)
        if line != u'':
            corpus.append(line)

for line in corpus:
    c = line.split('\t')

    node = c[0]
    pred = c[3]
    gold = c[2]

    rel_raw = c[1][1:]

    node_syn = wn.synset(node)
    node_lex = node_syn.lexname()

    gold_syn = wn.synset(gold)
    gold_lex = gold_syn.lexname()

    if rel_raw == 'hypernym':
        if '.v.' in node:
            rel = rel_raw + '_v'
        else: # '.n.' in node:
            rel = rel_raw + '_n'
    else: # 'instance_hypernym'
        rel = rel_raw

    if pred == '<unk>':
        pred_syn = None
        pred_lex = None
        is_gold_dev = False
        is_gold_wn = False
        lex_ident = False
        wup = 0.00

    else:
        pred_syn = wn.synset(pred)
        pred_lex = pred_syn.lexname()

        is_gold_dev = pred == gold  # needs revision (cases where multiple golds exist in WN18RR's train & dev set)
        is_gold_wn = pred_syn in node_syn.hypernyms() or pred_syn in node_syn.instance_hypernyms()
        lex_ident = pred_lex == gold_lex

        wup = pred_syn.wup_similarity(gold_syn)

        if wup is None:
            wup = 0.00

    with open(out_fname, 'a') as rerank_file:
        if rerank_file is not None:
            rerank_file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.5f}\n' \
                              .format(rel,
                                      node, node_lex,
                                      pred, pred_lex,
                                      gold, gold_lex,
                                      is_gold_dev,
                                      is_gold_wn,
                                      lex_ident,
                                      wup))




