from nltk.corpus import wordnet as wn
import re
import sys

in_fname = sys.argv[1]
out_fname = sys.argv[2]

# ------------------------------- #
# NB. Output column order:
# ------------------------------- #
#   relation
#
#   node (hyponym)
#   node_lexname
#
#   gold (hypernym)
#   gold_lexname
#
#   pred (hypernym)
#   pred_lexname
#
#   is_gold_wn18rr_dev
#   is_gold_wordnet
#
#   lexname identity (wn18rr_dev)
#   lexname identity (wordnet)
#
#   Wu & Palmer similarity (wn18rr_dev)
#   Wu & Palmer similarity (wordnet)
# ------------------------------- #

def pred_gold_ident(pred_syn, gold_syn):
    is_gold_dev = pred_syn == gold_syn  # needs revision (cases where multiple golds exist in WN18RR's train & dev set)
    is_gold_wn = pred_syn in node_syn.hypernyms() or pred_syn in node_syn.instance_hypernyms()
    return is_gold_dev, is_gold_wn


def lex_ident(node_syn, pred_syn, gold_syn):
    pred_lex = pred_syn.lexname()
    gold_lex = gold_syn.lexname()

    lex_ident_dev = pred_lex == gold_lex
    lex_ident_wn = pred_lex in [x.lexname() for x in node_syn.instance_hypernyms()] \
                   or pred_lex in [x.lexname() for x in node_syn.hypernyms()]
    return lex_ident_dev, lex_ident_wn, pred_lex, gold_lex


def wup_score(pred_syn, gold_syn):
    if pred_syn == gold_syn:
        wup_dev = 1.00
    else:
        wup_dev = pred_syn.wup_similarity(gold_syn)
        if wup_dev is None:
            wup_dev = 0.00

    if pred_syn in node_syn.hypernyms() or pred_syn in node_syn.instance_hypernyms():
        wup_wn_max = 1.00
    else:
        wup_wn = [pred_syn.wup_similarity(x) for x in node_syn.instance_hypernyms()]
        wup_wn.extend([pred_syn.wup_similarity(x) for x in node_syn.hypernyms()])
        if len(wup_wn) == 0 or all(x is None for x in wup_wn):
            wup_wn_max = 0.00
        else:
            wup_wn_max = max(wup_wn)
    return wup_dev, wup_wn_max


with open(in_fname, 'r') as f:
    corpus = []
    for line in f:
        line = re.sub('\n', '', line)
        if line != u'':
            corpus.append(line)

with open(out_fname, 'w') as rerank_file:
    rerank_file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n' \
                      .format('relation', 'node', 'gold', 'gold_lex', 'pred', 'pred_lex',
                              'is_gold_dev', 'is_gold_wn',
                              'lex_ident_dev', 'lex_ident_wn',
                              'wup_dev', 'wup_wn_max'))

    for line in corpus:
        node, rel_raw, gold, pred = line.split('\t')
        rel_raw = rel_raw[1:]

        # Load node and gold synsets from WordNet in NLTK
        node_syn = wn.synset(node)
        gold_syn = wn.synset(gold)

        # If the line has no prediction at all: pred = '<unk>'
        if pred == '':
            pred = '<unk>'

        # Relation type definition
        if rel_raw == 'hypernym':
            if '.v.' in node:
                rel = rel_raw + '_v'
            else:  # '.n.' in node:
                rel = rel_raw + '_n'
        else:  # 'instance_hypernym'
            rel = rel_raw

        # If predicted <unk> (or no prediction at all)
        if pred == '<unk>':
            pred_syn = None
            pred_lex = None
            is_gold_dev = False
            is_gold_wn = False
            lex_ident_dev = False
            lex_ident_wn = False

            wup_dev = 0.00
            wup_wn_max = 0.00

        # Most other cases where we have a predicted hypernym from our model
        else:
            pred_syn = wn.synset(pred)

            # Prediction correctness
            is_gold_dev, is_gold_wn = pred_gold_ident(pred_syn, gold_syn)

            # Lexname identity
            lex_ident_dev, lex_ident_wn, pred_lex, gold_lex = lex_ident(node_syn, pred_syn, gold_syn)

            # Wu & Palmer score
            wup_dev, wup_wn_max = wup_score(pred_syn, gold_syn)

        rerank_file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.5f}\t{:.5f}\n' \
                          .format(rel, node, gold, gold_lex, pred, pred_lex,
                                  is_gold_dev, is_gold_wn,
                                  lex_ident_dev, lex_ident_wn,
                                  wup_dev, wup_wn_max))
