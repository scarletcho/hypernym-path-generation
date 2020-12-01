import numpy as np

import dataloader

def decode_sequence(encoder_model, decoder_model, target_i2c, target_w2i, input_seq, decoding_json):
    """ GREEDY DECODER for now
    """
    try:
        decoder_model.get_layer(name='attention')
        has_attention = True
    except:
        has_attention = False

    D = dataloader.load_dict_from_json(decoding_json)
    #D = dataloader.load_dict_from_json('word_models/word_decoding.json')
    max_decoder_seq_length = D['max_decoder_seq_length']
    ##num_unique_target_chars = 4 #don't need this
    #states_value = encoder_model.predict(input_seq)
    vals = encoder_model.predict(input_seq)

    if has_attention:
        states_value = vals[1:]
        enc_outs = vals[0]
    else:
        states_value = vals

    stop_condition = False
    decoded_sentence = ''
    char_ind = target_w2i['START_']
    target_seq = np.zeros((1,1))
    target_seq[0,0] = char_ind
    cumlogprob = 0.0

    def predict_next_char(char_ind, states_value, cumlogprob):

        # Populate the first word of target sequence with the start word.
#        target_seq = np.zeros((1, 1, num_unique_target_chars))
#        target_seq[0,0, char_ind] = 1.0

        target_seq = np.zeros((1,1))
        target_seq[0,0] = char_ind

        if has_attention:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value + [enc_outs])
        else:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        states_value = [h, c]
        char_ind = np.argmax(output_tokens[0, -1, :])

        prob = output_tokens[0,-1, char_ind]
        logprob = np.log(prob)
        cumlogprob = cumlogprob + logprob

        return char_ind, states_value, cumlogprob

    #char_ind, states_value, cumlogprob = predict_next_char(char_ind, states_value, 0.0)
    #decoded_sentence += target_i2c[char_ind]

    while not stop_condition:
        char_ind, states_value, cumlogprob = predict_next_char(char_ind, states_value, cumlogprob)
        sampled_char = target_i2c[char_ind]
        decoded_sentence += sampled_char + " "

        # Exit condition: either hit max length or find ending word.
        if sampled_char == '_END': #or len(decoded_sentence.strip().split(" ")) > max_decoder_seq_length):
            stop_condition = True



    return decoded_sentence, cumlogprob

