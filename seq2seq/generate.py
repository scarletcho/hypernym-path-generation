import keras
import sys

import dataloader
from wordseq2seq import make_encoder, make_decoder
from greedy_decode import decode_sequence

if __name__ == '__main__':
    """ Example:

    python generate.py word_models/word_encoding.json word_models/word_decoding.json word_models/weights.03-1.67.h5 hyp_data2/src-val.txt pred.txt

    """
    ENCODING_INFO = sys.argv[1]
    DECODING_INFO = sys.argv[2]
    MODEL = sys.argv[3]
    SRC_LIST = sys.argv[4] # file with list of things to translate
    PRED_FILENAME = sys.argv[5] # where to write results

    #ENCODING_INFO = 'word_models/word_encoding.json'
    #DECODING_INFO = 'word_models/word_decoding.json'
    #MODEL = 'word_models/weights.03-1.67.h5'
    #SRC_LIST = 'hyp_data2/src-val.txt'
    #PRED_FILENAME = 'tmp-out2.txt'

    has_attention = True

    d_src = dataloader.load_dict_from_json(ENCODING_INFO)
    max_source_sentence = d_src['max_encoder_seq_length']
    input_w2i = d_src['input_c2i']
    num_unique_input_chars = len(input_w2i)

    d_tgt = dataloader.load_dict_from_json(DECODING_INFO)
    #max_target_sentence = d_tgt['max_decoder_seq_length']
    target_i2w = d_tgt['target_i2c']
    target_i2w = dict((int(i),word) for i,word in target_i2w.items() )
    target_w2i = dict((word, i) for i,word in target_i2w.items())

    with open(SRC_LIST,'r',encoding='utf8') as fd:
        t = fd.readlines()
    t = [i.strip() for i in t]

    encoded_inputs = dataloader.encode_texts(t, input_w2i, max_source_sentence)

    model = keras.models.load_model(MODEL)
    encoder_model = make_encoder(model, num_unique_input_chars, has_attention)
    decoder_model = make_decoder(model, has_attention)

    generated = []
    for ii in range(encoded_inputs.shape[0]):
        input_seq = encoded_inputs[ii:ii+1]
        d2,c2 = decode_sequence(encoder_model, decoder_model, target_i2w, target_w2i, input_seq, DECODING_INFO)
        d2s = d2.strip(' _END')
        generated.append(d2s)
        #print(ii,d2s)

    with open(PRED_FILENAME,'w',encoding='utf8') as fd:
        for g in generated:
            fd.write(g+'\n')

    # d2,c2 =  decode_sequence(encoder2, decoder2, target_i2w, target_w2i, input_seq)

