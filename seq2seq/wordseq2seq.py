"""A word-level Sequence to sequence model in Keras.

Adapted from:
- https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
- https://wanasit.github.io/english-to-katakana-using-sequence-to-sequence-in-keras.html
- https://github.com/devm2024/nmt_keras/blob/master/base.ipynb

Summary
-------
- An encoder LSTM turns input sequences to 2 state vectors
    (we keep the last LSTM state and discard the outputs).
- A decoder LSTM is trained to turn the target sequences into
    the same sequence but offset by one timestep in the future,
    a training process called "teacher forcing" in this context.
    Is uses as initial state the state vectors from the encoder.
    Effectively, the decoder learns to generate `targets[t+1...]`
    given `targets[...t]`, conditioned on the input sequence.
- In inference mode, when we want to decode unknown input sequences:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence word)
    - Feed the state vectors and 1-char target sequence
        to the decoder to produce predictions for the next word
    - Sample the next word using these predictions (simply use argmax).
    - Append the sampled word to the target sequence
    - Repeat until we generate the end-of-sequence word.

References
----------
- Sequence to Sequence Learning with Neural Networks
    https://arxiv.org/abs/1409.3215
- Learning Phrase Representations using
    RNN Encoder-Decoder for Statistical Machine Translation
    https://arxiv.org/abs/1406.1078

For more ideas on hyperparameter search:

- Massive Exploration of Neural Machine Translation Architectures, 2017
  https://arxiv.org/pdf/1703.03906.pdf

"""
from __future__ import print_function
import os
import random
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Bidirectional, Concatenate, Embedding
from keras.layers import Activation, dot, concatenate, TimeDistributed
import pandas as pd
import argparse

import dataloader

# TODO Some questions:
# -- Does it make sense to apply dropout to both encoder and decoder?


def make_model(num_unique_input_chars,
               num_unique_target_chars,
               latent_dim,
               dropout_encoder,
               dropout_decoder,
               rec_dropout,
               has_attention=False,
               src_embedding_matrix=None,
               tgt_embedding_matrix=None,
               trainable_src_emb=False,
               trainable_tgt_emb=False):
    """Create the LSTM encoder-decoder model."""

    if src_embedding_matrix is not None:
        src_embedding_matrix = [src_embedding_matrix]
    if tgt_embedding_matrix is not None:
        tgt_embedding_matrix = [tgt_embedding_matrix]

    # ENCODER ARCHITECTURE
    ######################

    encoder_raw_inputs = Input(shape=(None,)) 

    #encoder_inputs = Embedding(num_unique_input_chars, embedding_size)(encoder_raw_inputs)
    denc = Embedding(num_unique_input_chars,
                     embedding_size,
                     name = 'encoder_embedding',
                     weights = src_embedding_matrix, #[embedding_matrix]
                     trainable = trainable_src_emb
                     )

    encoder_inputs = denc(encoder_raw_inputs)
#    encoder_inputs = Input(shape=(None, num_unique_input_chars),
#                           name='input_encoder')

    lstm_layer = LSTM(latent_dim,
                      name='lstm_encoder',
                      dropout=dropout_encoder,
                      recurrent_dropout=rec_dropout,
                      return_state=True,
                      return_sequences=True)

    enc_outs, state_h, state_c = lstm_layer(encoder_inputs)

    encoder_states = [state_h, state_c]

    # DECODER ARCHITECTURE -- use `encoder_states` as initial state.
    ######################

#    decoder_inputs = Input(shape=(None, num_unique_target_chars),
#                           name='input_decoder')
    decoder_raw_inputs = Input(shape=(None,), name='input_decoder')
    dex = Embedding(num_unique_target_chars,
                    embedding_size,
                    name='decoder_embedding',
                    weights = tgt_embedding_matrix,
                    trainable = trainable_tgt_emb
                    )

    decoder_inputs = dex(decoder_raw_inputs) #final_dex

    # The decoder will return both full output sequences and internal states.
    # Return states will be used in inference, not in training.

    decoder_lstm = LSTM(latent_dim,
                        name='lstm_decoder',
                        dropout=dropout_decoder,
                        recurrent_dropout=rec_dropout,
                        return_sequences=True,
                        return_state=True)

    decoder_lstm_outputs, _, _ = decoder_lstm(decoder_inputs,
                                              initial_state=encoder_states)

    if has_attention:
        # The following equation numbers are from Luong et al., section 3.1.
        score = dot([decoder_lstm_outputs, enc_outs], axes=[2, 2])  # Eq. (7)
        # The output is a rank-3 tensor, where first dim= number of instances,
        # The second dim is max_target_sentence_length, and the third dim is
        # max_source_sentence_length. Entry i,j,k corresponds to instance i, time-step
        # j of the decoder, timestep k of the encoder.
        attention = Activation('softmax', name='attention')(score)  # Eq. (7)
        # Row i,j,: are the weights for the ith sample, for the jth timestep of
        # the decoder. The softmax normalized them. There are
        # max_source_sentence_length weights in the row.
        context = dot([attention, enc_outs], axes=[2, 1])
        # Row i,j,: is the context vector for instance i, decoder timestep j,
        # ie. weighted average (using attention weights) of the encoder hidden
        # states.

        # Eq. (5):
        decoder_combined = concatenate([context, decoder_lstm_outputs])

        output = TimeDistributed(Dense(latent_dim,
                                       activation="tanh"))(decoder_combined)

        # Eq. (6): the conditional probabilities
        decoder_outputs = TimeDistributed(Dense(num_unique_target_chars,
                                                activation="softmax"))(output)

        #model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model = Model([encoder_raw_inputs, decoder_raw_inputs], decoder_outputs)
    else:
        decoder_dense = Dense(num_unique_target_chars,
                              activation='softmax')
        decoder_outputs = decoder_dense(decoder_lstm_outputs)
#        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model = Model([encoder_raw_inputs, decoder_raw_inputs], decoder_outputs)

    return model

#NOTE: later remove the num_unique_input_chars ; it is not being used.
def make_encoder(model, num_unique_input_chars, has_attention=False):
    """."""
#    encoder_inputs = model.input[0]
    encoder_raw_inputs = model.input[0]
    #encoder_inputs = Embedding(num_unique_input_chars, embedding_size)(encoder_raw_inputs)
    denc = model.get_layer('encoder_embedding')
    encoder_inputs = denc(encoder_raw_inputs)

    outputs = model.get_layer(name='lstm_encoder').output

    encoder_states = outputs[1:]
    out = outputs[0]

    if has_attention:
        encoder_out = [out]+encoder_states
    else:
        encoder_out = encoder_states

#    encoder_model = Model(encoder_inputs, encoder_out)
    encoder_model = Model(encoder_raw_inputs, encoder_out)

    return encoder_model


def make_decoder(model, has_attention=False):
    """."""
    latent_dim = model.get_layer(name='lstm_encoder').output_shape[0][-1]

#    num_unique_target_chars = model.get_layer(name='input_decoder').input_shape[2]

    decoder_states_inputs = [Input(shape=(latent_dim,)),
                             Input(shape=(latent_dim,))]

#    decoder_inputs2 = Input(shape=(None, num_unique_target_chars))
    decoder_raw_inputs2 = Input(shape=(None,))
    dex = model.get_layer('decoder_embedding')
    decoder_inputs2 = dex(decoder_raw_inputs2) # final_dex2

    decoder_lstm = model.get_layer('lstm_decoder')
    decoder_lstm_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs2,
        initial_state=decoder_states_inputs)

    decoder_states = [state_h, state_c]

    if has_attention:
        enc_outs = Input(shape=(None, latent_dim))
        score = dot([decoder_lstm_outputs, enc_outs], axes=[2, 2])
        attention = Activation('softmax', name='attention')(score)
        context = dot([attention, enc_outs], axes=[2, 1])
        decoder_combined = concatenate([context, decoder_lstm_outputs])

        dense_0 = model.layers[-2]
        dense_1 = model.layers[-1]
        output = dense_0(decoder_combined)
        decoder_outputs = dense_1(output)

        decoder_model = Model([decoder_raw_inputs2] + decoder_states_inputs + [enc_outs],
                              [decoder_outputs] + decoder_states)
#        decoder_model = Model([decoder_inputs2] + decoder_states_inputs + [enc_outs],
#                              [decoder_outputs] + decoder_states)
    else:
        decoder_dense = model.layers[-1]
        decoder_outputs = decoder_dense(decoder_lstm_outputs)
#        decoder_model = Model([decoder_inputs2] + decoder_states_inputs,
#                              [decoder_outputs] + decoder_states)
        decoder_model = Model([decoder_raw_inputs2] + decoder_states_inputs,
                              [decoder_outputs] + decoder_states)

    return decoder_model


def fit_model(model,
              clipnorm,
              learning_rate,
              optimizer_name,
              encoder_input_train,
              decoder_input_train,
              decoder_target_train,
              encoder_input_val,
              decoder_input_val,
              decoder_target_val,
              save_filename,
              save_checkpoint_epochs):
    """."""
    if optimizer_name == 'adam':
        epsilon = 1e-07 #8e-07  # The keras default value was 1e-07 (K.epsilon())
        if clipnorm is not None:
            opt = keras.optimizers.Adam(clipnorm=clipnorm,
                                        lr=learning_rate,
                                        epsilon=epsilon)
        else:
            opt = keras.optimizers.Adam(epsilon=epsilon,
                                        lr=learning_rate)
    else:
        raise NotImplementedError("Use optimizer_name = 'adam' for now.")

    model.compile(optimizer=opt,
                  metrics=['accuracy'],
                  loss='categorical_crossentropy')

    #early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    checkpointer = ModelCheckpoint(filepath='word_models/weights.{epoch:02d}.h5',
                                   verbose=0,
                                   save_weights_only=False,
                                   save_best_only=False,
                                   period=save_checkpoint_epochs)

    #NOTE: to feed it more data, think about replacing it with fit_generator
    # see https://github.com/keras-team/keras/issues/2708
    history = model.fit([encoder_input_train, decoder_input_train],
                        decoder_target_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        #callbacks=[early_stopping, checkpointer],
                        callbacks = [checkpointer],
                        validation_data = ([encoder_input_val, decoder_input_val],
                                            decoder_target_val)
                        )
                        ##validation_split=0.2)

    model.save('word_models/'+save_filename)
    return history


if __name__ == "__main__":
   # parser = argparse.ArgumentParser(description='Train a word-level LSTM seq2seq model.')
    parser = argparse.ArgumentParser(description='Train a word-level LSTM seq2seq model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train_src', type=str, help='File with source data for training.')
    parser.add_argument('--train_tgt', type=str, help='File with target data for training.')
    parser.add_argument('--valid_src', type=str, help='File with source data for validation.')
    parser.add_argument('--valid_tgt', type=str, help='File with target data for validation.')
    parser.add_argument('--test_src', type=str, help='File with source data for testing.')
    parser.add_argument('--test_tgt', type=str, help='File with target data for testing.')

    parser.add_argument('--epochs', type=int, help='Number of epochs to train.')

    parser.add_argument('--save_checkpoint_epochs', type=int, help='Save checkpoint every N epochs.', default=5)
    parser.add_argument('--num_samples_train', type=int, help='Number of training samples. Use 0 to use all of them.', default=0)
    parser.add_argument('--num_samples_val', type=int, help='Number of validation samples. Use 0 to use all of them.', default=0)
    parser.add_argument('--emb_file_enc', type=str, help='File with word embeddings for encoder. Use None if you do not wish to use pretrained embeddings.', default="embs/ft-embs-all-lower.vec")
    parser.add_argument('--emb_file_dec', type=str, help='File with word embeddings for decoder. Use None if you do not wish to use pretrained embeddings.', default="embs/ft-embs-all-lower.vec")
    parser.add_argument('--word_vec_size', type=int, help='Word embedding dimension.', default=300)
    parser.add_argument('--latent_dim', type=int, help='Number of hidden units for LSTM.', default=256)
    parser.add_argument('--dropout_encoder', type=float, help='Fraction of units to dropout for encoder.', default=0.3)
    parser.add_argument('--dropout_decoder', type=float, help='Fraction of units to dropout for decoder.', default=0.3)
    parser.add_argument('--batch_size', type=int, help='Batch size.', default=256)
    parser.add_argument('--learning_rate', type=float, help='Learning rate.', default=0.001)
    parser.add_argument("--trainable_src_emb", help="Train source embedding. Will be frozen by default.",
                        action="store_true")
    parser.add_argument("--trainable_tgt_emb", help="Train target embedding. Will be frozen by default.",
                        action="store_true")
    parser.add_argument("--attention", help="Turn on Luong attention. Will be off by default.",
                        action="store_true")

    args = parser.parse_args()
    has_attention = args.attention
    embedding_size = args.word_vec_size #300 #100 #300
    latent_dim = args.latent_dim #256  # 256 #256  # Could try 512, but slower
    dropout_encoder = args.dropout_encoder #0.3 # 0.3
    dropout_decoder = args.dropout_decoder
    batch_size = args.batch_size #256 #128 #256  # See Neishi et al, 'A Bag of Useful Tricks' (2017)
    learning_rate = args.learning_rate #0.001 #0.0005  # 0.0001 # Keras default was 0.001
    epochs = args.epochs #3#50 #25# 200  # 200 # 100 was too low
    save_checkpoint_epochs = args.save_checkpoint_epochs
    MODEL_NAME = 's2s.h5'
    clipnorm = None
    rec_dropout = 0.0
    optimizer_name = 'adam'
    training_data_path_src = args.train_src
    training_data_path_tgt = args.train_tgt
    validation_data_path_src = args.valid_src
    validation_data_path_tgt = args.valid_tgt
    test_data_path_src = args.test_src
    test_data_path_tgt = args.test_tgt
    trainable_src_emb = args.trainable_src_emb
    trainable_tgt_emb = args.trainable_tgt_emb
    src_emb_file = args.emb_file_enc
    tgt_emb_file = args.emb_file_dec
    if src_emb_file=='None':
        src_emb_file = None
    if tgt_emb_file=='None':
        tgt_emb_file = None
    num_samples_train = args.num_samples_train
    num_samples_val = args.num_samples_val
    if num_samples_train == 0:
        num_samples_train = None
    if num_samples_val == 0:
        num_samples_val = None

    #training_data_path_src = "hyp_data2/src-train.txt"
    #training_data_path_tgt = "hyp_data2/tgt-train.txt"
    #validation_data_path_src = "hyp_data2/src-val.txt"
    #validation_data_path_tgt = "hyp_data2/tgt-val.txt"
    #src_emb_file = "embs/ft-embs-all-lower.vec" # Use None to not use pretrained
    #tgt_emb_file = "embs/ft-embs-all-lower.vec"
    #trainable_src_emb=False
    #trainable_tgt_emb=False
    #num_samples_train = 15000 #0 #10 #3000 #20000 #35000  # 50000 # 250000 is too large; 50000 was OK.
    #num_samples_val = None

    train_data = dataloader.load_data((training_data_path_src, training_data_path_tgt), numlines=num_samples_train)
    val_data = dataloader.load_data((validation_data_path_src, validation_data_path_tgt), numlines=num_samples_val)
    test_data = dataloader.load_data((test_data_path_src, test_data_path_tgt), numlines=num_samples_val)

    #train_and_val = pd.concat([train_data, val_data], ignore_index=True)
    train_and_val_and_test = pd.concat([train_data, val_data, test_data], ignore_index=True)

    max_source_sentence_length, max_target_sentence_length = dataloader.get_max_sentence_lengths(train_and_val_and_test)

    input_w2i, target_w2i = dataloader.prepare_data(train_and_val_and_test)
    target_i2w = dict((i, word) for word,i in target_w2i.items())
    # Only need this if using pretrained embeddings for encoder.
    input_i2w = dict((i, word) for word,i in input_w2i.items())

    num_unique_input_chars = len(input_w2i)
    num_unique_target_chars = len(target_w2i)

    #NOTE: notice I didn't use the final (test) part to create these arrays.
    encoder_input_data, decoder_input_data, decoder_target_data = dataloader.get_encoder_and_decoder_arrays(input_w2i, target_w2i, max_source_sentence_length, max_target_sentence_length, train_and_val_and_test[:len(train_data)+len(val_data)])

    encoder_input_train = encoder_input_data[:len(train_data), :]
    encoder_input_val = encoder_input_data[len(train_data):, :]

    decoder_input_train = decoder_input_data[:len(train_data), :]
    decoder_input_val = decoder_input_data[len(train_data):, :]

    decoder_target_train = decoder_target_data[:len(train_data),:]
    decoder_target_val = decoder_target_data[len(train_data):, :]

    char_encoding = {'max_encoder_seq_length': max_source_sentence_length,
                     'input_c2i': input_w2i}
    char_decoding = {'max_decoder_seq_length': max_target_sentence_length,
                     'target_i2c': target_i2w}

    if not os.path.exists('./word_models'):
        os.makedirs('./word_models')

    dataloader.write_dict_to_json('word_models/word_decoding.json', char_decoding)
    dataloader.write_dict_to_json('word_models/word_encoding.json', char_encoding)

    if src_emb_file is not None:
        src_w2v = dataloader.load_embs(src_emb_file)
        emb_matrix_src, src_OOV = dataloader.get_emb_matrix(input_i2w, src_w2v, embedding_size, oov_filename='word_models/src_oov.txt')
    else:
        emb_matrix_src = None

    if tgt_emb_file is not None:
        tgt_w2v = dataloader.load_embs(tgt_emb_file)
        emb_matrix_tgt, tgt_OOV = dataloader.get_emb_matrix(target_i2w, tgt_w2v, embedding_size, oov_filename='word_models/tgt_oov.txt')
    else:
        emb_matrix_tgt = None

    model = make_model(num_unique_input_chars,
                       num_unique_target_chars,
                       latent_dim,
                       dropout_encoder,
                       dropout_decoder,
                       rec_dropout,
                       has_attention=has_attention,
                       src_embedding_matrix=emb_matrix_src,
                       tgt_embedding_matrix=emb_matrix_tgt,
                       trainable_src_emb=trainable_src_emb,
                       trainable_tgt_emb=trainable_tgt_emb
                       )

    history = fit_model(model, clipnorm, learning_rate,
                        optimizer_name,
                        encoder_input_train,
                        decoder_input_train,
                        decoder_target_train,
                        encoder_input_val,
                        decoder_input_val,
                        decoder_target_val,
                        MODEL_NAME,
                        save_checkpoint_epochs)

    decoder_model = make_decoder(model, has_attention=has_attention)

    encoder_model = make_encoder(model, num_unique_input_chars, has_attention=has_attention)


#NOTE: to load the model from disk:
# model2 = keras.models.load_model(modelpath)
# encoder2 = make_encoder(model2, num_unique_input_chars, has_attention)
# decoder2 = make_decoder(model2, has_attention)
# input_seq = encoder_input_train[ii:ii+1]
# d2,c2 =  decode_sequence(encoder2, decoder2, target_i2w, target_w2i, input_seq)

#from greedy_decode import decode_sequence
#
#for seq_index in range(10):
#    input_seq = encoder_input_train[seq_index: seq_index + 1]
#    #decoded_sentence = decode_sequence(input_seq)
#    d2,c2 = decode_sequence(encoder_model, decoder_model, target_i2w, target_w2i, input_seq); print('-')
#    print(train_data.src[seq_index])
#    print('real: ',train_data.tgt[seq_index])
#    print('pred: '+'START_', d2)

