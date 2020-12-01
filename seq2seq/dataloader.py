import pandas as pd
import re
import string
import numpy as np
import json

def get_emb_matrix(target_i2w, w2v, EMB_DIM, unk_size=0.25, oov_filename=None):
    emb_mat = np.zeros((len(target_i2w), EMB_DIM))
    notfound = []

    for i in target_i2w.keys():
        if i!=0:
            if target_i2w[i] in w2v:
                emb_mat[i] = w2v[target_i2w[i]]
            else:
                notfound.append(target_i2w[i])
                #print(target_i2w[i], "not there!")
                emb_mat[i] = np.random.uniform(-unk_size, unk_size, EMB_DIM)

    if oov_filename is not None:
        with open(oov_filename, 'w') as fd:
            for i in notfound:
                fd.write(i+'\n')

    return emb_mat, notfound


def load_embs(embfile):
    with open(embfile,'r',encoding='utf8') as fd:
        t = fd.readlines()
    t = [i.strip() for i in t]
    t = [i.split(' ') for i in t]
    words = [i[0] for i in t]
    vecs = [i[1:] for i in t]
    vecs = [np.array([float(i) for i in vec]) for vec in vecs]
    D = dict(zip(words, vecs))
    return D

def write_dict_to_json(filename, D):
    print("Writing dictionary to "+filename)
    with open(filename, 'w', encoding='utf8') as f:
        f.write(json.dumps(D, sort_keys=True, indent=4))

def load_dict_from_json(filename):
    with open(filename,'r',encoding='utf8') as f:
        data = json.loads(f.read())
    return data


def load_data(filename, numlines=None, optional_processing=False):#50000):
    """ filename: either str (name of file; the file must be a list of pairs
        in the form src \t tgt) or tuple (src, tgt).

        If numlines is None, load all the data.

        If optional_processing is True:
            - lowercase everything
            - remove punctuation

    """
    #filename = 'data-text/fra.txt'
    if isinstance(filename,str):
        lines = pd.read_table(filename, names=['src', 'tgt'])
        #print("Number of samples used from "+filename+": ",str(len(lines)) )
    elif isinstance(filename,tuple):
        s,t = load_two_files(*filename) # first is src, second is tgt.
        lines = pd.DataFrame({'src':s, 'tgt':t})
        #print("Number of samples used from "+", ".join(filename)+": ",str(len(lines)) )
    else:
        raise ValueError("Must be either name of the file or a tuple of filenames (src, tgt).")

    if numlines is not None:
        lines = lines[:numlines]


    if optional_processing:
        lines.src=lines.src.apply(lambda x: x.lower())
        lines.tgt=lines.tgt.apply(lambda x: x.lower())
        #lines.src=lines.src.apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' COMMA', x))
        #lines.tgt=lines.tgt.apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' COMMA', x))
        exclude = set(string.punctuation)
        lines.src=lines.src.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
        lines.tgt=lines.tgt.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

    lines.tgt = lines.tgt.apply(lambda x : 'START_ '+ x + ' _END')

    return lines


def load_two_files(src_file, tgt_file):
    with open(src_file,'r', encoding='utf8') as fd:
        src = fd.readlines()
    src = [i.strip() for i in src]

    with open(tgt_file,'r', encoding='utf8') as fd:
        tgt = fd.readlines()
    tgt = [i.strip() for i in tgt]

    return src, tgt

def get_max_sentence_lengths(lines):
    max_target_sentence = max([len(i.split()) for i in lines.tgt])
    max_source_sentence = max([len(i.split()) for i in lines.src])
    print("Max sequence length for inputs: ", max_source_sentence)
    print("Max sequence length for outputs: ", max_target_sentence)
    return max_source_sentence, max_target_sentence


def prepare_data_shared(lines):
    all_src_words=set()
    for src in lines.src:
        for word in src.split():
            if word not in all_src_words:
                all_src_words.add(word)

    all_tgt_words=set()
    for tgt in lines.tgt:
        for word in tgt.split():
            if word not in all_tgt_words:
                all_tgt_words.add(word)

    all_words = all_src_words | all_tgt_words
    all_words = all_words - {'START_'}
    #all_tgt_words = all_tgt_words - {'START_'}

    #input_words = sorted(list(all_src_words))
    target_words = ['START_']+sorted(list(all_words)) # want 'START_' to be the first

    #NOTE: want the first entry (0th) to correspond to the start symbol
    #input_w2i = dict(
    #    [(word, i) for i, word in enumerate(input_words)])
    target_w2i = dict(
        [(word, i) for i, word in enumerate(target_words)])

    input_w2i = target_w2i

    print("Target vocab size: ", len(target_w2i))
    print("Source vocab size: ", len(input_w2i))
    return input_w2i, target_w2i


def prepare_data(lines):
    all_src_words=set()
    for src in lines.src:
        for word in src.split():
            if word not in all_src_words:
                all_src_words.add(word)

    all_tgt_words=set()
    for tgt in lines.tgt:
        for word in tgt.split():
            if word not in all_tgt_words:
                all_tgt_words.add(word)
    all_tgt_words = all_tgt_words - {'START_'}

    input_words = sorted(list(all_src_words))
    target_words = ['START_']+sorted(list(all_tgt_words)) # want 'START_' to be the first

    #NOTE: want the first entry (0th) to correspond to the start symbol
    input_w2i = dict(
        [(word, i) for i, word in enumerate(input_words)])
    target_w2i = dict(
        [(word, i) for i, word in enumerate(target_words)])

    print("Source vocab size: ", len(input_w2i))
    print("Target vocab size: ", len(target_w2i))

    return input_w2i, target_w2i


def encode_texts(input_texts, input_w2i, max_source_sentence):

    ##max_source_sentence = max([len(i.split()) for i in input_texts])

    encoder_input = np.zeros(
        (len(input_texts), max_source_sentence),
        dtype='float32')

    for i, input_text in enumerate(input_texts):
        for t, word in enumerate(input_text.split()):
            encoder_input[i, t] = input_w2i[word] #TODO get keyerror now.. what about OOV?

    return encoder_input


def get_encoder_and_decoder_arrays(input_w2i, target_w2i, max_source_sentence, max_target_sentence, lines):

    source_i2w = dict((i, word) for word,i in input_w2i.items())
    target_i2w = dict((i, word) for word,i in target_w2i.items())

    num_decoder_tokens = len(target_w2i)

    encoder_input_data = np.zeros(
        (len(lines.src), max_source_sentence),
        dtype='float32')

    decoder_input_data = np.zeros(
        (len(lines.tgt), max_target_sentence),
        dtype='float32')

    decoder_target_data = np.zeros(
        (len(lines.tgt), max_target_sentence, num_decoder_tokens),
        dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(lines.src, lines.tgt)):
        for t, word in enumerate(input_text.split()):
            encoder_input_data[i, t] = input_w2i[word]

        for t, word in enumerate(target_text.split()):
            decoder_input_data[i, t] = target_w2i[word]
            if t > 0:
                # Teacher forcing.
                # Decoder_target_data is ahead of decoder_input_data by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_w2i[word]] = 1. # probability=1 on the known word.

    # sanity check
#    def pad(x,padlen,padval):
#        if len(x)<padlen:
#            return x+[padval]*(padlen-len(x))
#        else:
#            return x
#    for ii in range(len(lines.src)):
#        S = [source_i2w[x] for x in encoder_input_data[ii]]
#        vals = lines.src[ii].split()
#        P = pad(vals, max_source_sentence, source_i2w[0])
#        if S!=P:
#            print('bad encoder val: ', ii)
#    for ii in range(len(lines.tgt)):
#        S = [target_i2w[x] for x in decoder_input_data[ii]]
#        vals = lines.tgt[ii].split()
#        P = pad(vals, max_target_sentence, target_i2w[0])
#        if S!=P:
#            print('bad decoder val: ', ii)

    return encoder_input_data, decoder_input_data, decoder_target_data


#if __name__ == "__main__":
#    filename = 'data-test/fra.txt'
#    numlines = 40000#801
#    lines = load_data(filename,numlines=numlines)
#    input_token_index, target_token_index, max_source_sentence, max_target_sentence = prepare_data(lines)

#    encoder_input_data, decoder_input_data, decoder_target_data = get_encoder_and_decoder_arrays(input_token_index, target_token_index, max_source_sentence, max_target_sentence, lines)

