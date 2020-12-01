# keras_seq2seq
Word-level seq2seq model in keras

## Download the synset embeddings

These are located at https://github.com/scarletcho/hypernym-path-generation/releases/tag/v1.0 (file `ft-embs-all-lower.vec`). Move it to the `embs/` directory to use the synset embeddings as in the examples below.

## Hyperparameters

The hyperparameters used in our paper are the default values for arguments in `wordseq2seq.py`. We used the following number of epochs for hypo2hyper rev. Nouns: 45. Verbs: 33. Instance nouns: 60.

## Train the model

Example use: train for 70 epochs with training source and target pairs in files `src-train.txt` and `tgt-train.txt` (and similarly for validation).

```
python3 wordseq2seq.py \
--train_src hyp_data2_nodehyp/src-train.txt \
--train_tgt hyp_data2_nodehyp/tgt-train.txt \
--valid_src hyp_data2_nodehyp/src-val.txt \
--valid_tgt hyp_data2_nodehyp/tgt-val.txt \
--test_src hyp_data2_nodehyp/src-test.txt \
--test_tgt hyp_data2_nodehyp/tgt-test.txt \
--emb_file_enc embs/ft-embs-all-lower.vec \
--emb_file_dec embs/ft-embs-all-lower.vec \
--epochs 100 \
--attention \
```

Comments on the arguments:

- If you *don't* want to use Luong attention, remove the `--attention` flag.
- To only use the first N lines of training data: `--num_samples_train N`. To only use first N lines of validation data: `--num_samples_val N`.
- If you *don't* wish to use pretrained embeddings for encoder and/or decoder, use: `--emb_file_enc None` and `--emb_file_dec None`.
- Word embeddings will be frozen by default. To train them, use flag `--trainable_src_emb` and/or `--trainable_tgt_emb`
- Use `--save_checkpoint_epochs N` to save a checkpoint every N epochs.
- To see other command line arguments and default values, type:
 ```python wordseq2seq.py -h```

The trained model will be saved in directory `word_models` (if it already exists it will be overwritten).

## Generate translations

This is an example use that translates from `src-val.txt` and saves results to `pred.txt`:

```
python3 generate.py \
word_models/word_encoding.json \
word_models/word_decoding.json \
word_models/weights.60.h5 \
hyp_data2/src-val.txt \
pred.txt
```

The first two arguments are the encoding and decoding json files needed to load the data.  This is followed by one of the saved model .h5 files (each file contains the number of epochs in its name) and the file containing the source sentences to translate. The last argument is the name of the file to write results to.

## Evaluate translations

To evaluate on the validation set, for instance nouns, use the following (60 indicates the number of epochs, and 1 at the end indicates the predicted paths should be reversed):

```
python3 combine.py src-valid.txt tgt-valid.txt  pred-val-60.txt 60 instnouns val 1
. evaluate.sh y_results_60e_instnouns_val results_name_here
```

The first argument consists of source hyponyms; the second argument consists of target gold-truth WordNet paths and the third argument is the file with predicted paths.
The script will create a `.out.summary.txt` file with various scores, including H@1 and Wu&P.

