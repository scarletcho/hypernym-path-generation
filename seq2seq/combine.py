import sys

if __name__ == '__main__':

    src_file = sys.argv[1]
    tgt_file = sys.argv[2]
    pred_file = sys.argv[3]
    epochs = sys.argv[4]
    category = sys.argv[5] # verbs, nouns or instnouns
    split = sys.argv[6]    # val, or test

    #optional (if used 'reversed' version, hypernym is first, not last)
    try: # use 1 here if want reversed.
        reverse = sys.argv[7]
        reverse = bool(reverse)
    except:
        reverse=False

    if category in {'verbs', 'nouns'}:
        hyp_name = '_hypernym'
    elif category == 'instnouns':
        hyp_name = '_instance_hypernym'
    else:
        raise ValueError("Must be 'nouns', 'verbs', or 'instnouns' ")

    with open(pred_file, 'r') as fd:
        pred = fd.readlines()

    with open(src_file, 'r') as fd:
        srcval = fd.readlines()

    with open(tgt_file, 'r') as fd:
        tgtval = fd.readlines()

    pred = [i.strip() for i in pred]

    if reverse:
        pred = [i.split(' ')[0] for i in pred]
    else:
        pred = [i.split(' ')[-1] for i in pred]

    srcval = [i.strip() for i in srcval]
    tgtval = [i.strip() for i in tgtval]


    with open('y_results_'+epochs+'e_'+category+'_'+ split +'.txt','w') as fd:
        for ii,i in enumerate(pred):
            fd.write(srcval[ii]+'\t'+hyp_name+'\t'+tgtval[ii]+'\t'+pred[ii]+'\n' )

