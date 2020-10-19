import pandas as pd
import sys

model_name = sys.argv[1]
pred_path = sys.argv[2]
score_path = sys.argv[3]


def h_at_1_dev(df_sub):
    # H@1 score based on WN18RR dev set ('is_gold_dev')
    #   = if the prediction is among the gold hypernym(s) of the given hyponym in WN18RR dev set
    ig_dev_true_cnt = sum(df_sub.loc[:, 'is_gold_dev'] == True)
    ig_dev_false_cnt = sum(df_sub.loc[:, 'is_gold_dev'] == False)

    h_at_1_wn18rr_dev = ig_dev_true_cnt / (ig_dev_true_cnt + ig_dev_false_cnt) * 100
    return round(h_at_1_wn18rr_dev, 2)


def h_at_1_wordnet(df_sub):
    # H@1 score based on the entire WordNet ('is_gold_wn')
    #   = if the prediction is among the gold hypernym(s) of the given hyponym in the entire WordNet
    ig_wn_true_cnt = sum(df_sub.loc[:, 'is_gold_wn'] == True)
    ig_wn_false_cnt = sum(df_sub.loc[:, 'is_gold_wn'] == False)

    h_at_1_wn = ig_wn_true_cnt / (ig_wn_true_cnt + ig_wn_false_cnt) * 100
    return round(h_at_1_wn, 2)


def lex_ident_dev_accuracy(df_sub):
    true_cnt = sum(df_sub.loc[:, 'lex_ident_dev'] == True)
    false_cnt = sum(df_sub.loc[:, 'lex_ident_dev'] == False)

    lex_ident_dev_acc = true_cnt / (true_cnt + false_cnt) * 100
    return round(lex_ident_dev_acc, 2)


def lex_ident_wordnet_accuracy(df_sub):
    true_cnt = sum(df_sub.loc[:, 'lex_ident_wn'] == True)
    false_cnt = sum(df_sub.loc[:, 'lex_ident_wn'] == False)

    lex_ident_wn_acc = true_cnt / (true_cnt + false_cnt) * 100
    return round(lex_ident_wn_acc, 2)


def wu_and_palmer_dev_avg(df_sub):
    wup_dev = sum(df_sub.loc[:, 'wup_dev']) / len(df_sub) * 100
    return round(wup_dev, 2)


def wu_and_palmer_wordnet_max_avg(df_sub):
    wup_wn_max = sum(df_sub.loc[:, 'wup_wn_max']) / len(df_sub) * 100
    return round(wup_wn_max, 2)


# Read tab-separated prediction file as pandas dataframe
df = pd.read_csv(pred_path, sep='\t')
rels = df.relation.unique()

with open(score_path, 'w') as f:
    f.write("model name: {}\n".format(model_name))
    f.write("model prediction: {}\n\n".format(pred_path))
    for relation_i in rels:
        f.write("relation: {}\n".format(relation_i))

        df_i = df.loc[df.loc[:, "relation"] == relation_i, :]

        h1_dev = h_at_1_dev(df_i)
        h1_wn = h_at_1_wordnet(df_i)
        lex_dev = lex_ident_dev_accuracy(df_i)
        lex_wn = lex_ident_wordnet_accuracy(df_i)
        wup_dev = wu_and_palmer_dev_avg(df_i)
        wup_wn_max = wu_and_palmer_wordnet_max_avg(df_i)

        f.write("h@1_dev\th@1_wn\tlex_dev\tlex_wn\twup_dev\twup_wn_max\n")
        f.write("{}\t{}\t{}\t{}\t{}\t{}\n\n".format(h1_dev, h1_wn,
                                                    lex_dev, lex_wn,
                                                    wup_dev, wup_wn_max))
        print(h1_dev, h1_wn, lex_dev, lex_wn, wup_dev, wup_wn_max)
