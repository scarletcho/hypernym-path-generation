# PathEncoder
## Dependency
- python 3.6+ (For default ordered dictionary and f-string)
- pytorch 1.0+  
- numpy  
- tqdm  
## Usage

Please call 
```bash
>>> python3 main.py -m PathEncoder --train $train_file --dev $dev_file --query $query_file --embeddings $embedding  --lr $LEARNING_RATE --valid --gamma $GAMMA
```
to train and test the model.  

### Arguments:

Args | Function | Note
------------ | ------------- | -----------
-m or --model $model    | Select which model to use.  *Possible $model: {PathEncoder, PathClassifier, NN, baseline}.* | PathEncoder is the model I reported in the slide. PathClassifier is the original path/hypo classifier model. NN takes only hyper/hypo. Baseline is just Nearest Neighbor Search.
--gamma $GAMMA    |  Assign value to the hyperparameter used in a margin loss   |  Functions only when using the PathEncoder model. 

### Other posible arguments:

Args | Function | Default
------------ | -------------  | -------
--neg   |  Enable negative examples for training. | False
--cuda  |  Enable CUDA. | False
--save_model   | Save the model and optimizer parameters. | False
--save_best    | Only save the best model parameters and query ouput. | False
--drop_out | Set dropout rate. | 0
--hit $hit | Set the number of ouputs per instance. | 10

### Example Usage:
```bash
>>> python3 main.py --train data/split/wn18rr_adjust/neg/train_verb_path_18rr_12r.r --dev data/split/wn18rr_adjust/neg/valid_verb_path_18rr_12r.r --query data/verb_wn18_path --neg --gamma 0.2 --lr 0.0001 -e 80 --valid --cuda 
```
which takes the default 300-dimension 'ft-embs-all-lower.vec' emebddings and train/dev on the verb dataset.

To train/dev on noun/instance:
replace all 'verb' in the command with 'noun' or 'inst'.



## Output
Ouput: Text files with predicted path for each instance in valid_raw.path where each line of valid_raw.path corresponds to a path of a word in the verb_wn18_path dataset for path query.

**NOTE: EACH epoch will generate a different ouput file.**

To get the formated output summaries and scores, please call
```bash
>>> python3 get_output.py $output_folder $hits
```

This will generate a .summary file for each model (with a particular hyperparameter settings) in the $output_folder.

The summary file is in the following format:

Metric | Best Score | @Epoch | Scores of other metircs on this epoch \[Metric_1, ..., Metric_5 \]
------------ | ------------- | -----  | ------- 
Metric_1 | Metric_1 Best | best epoch | other scores
... | ... | ... | ...
Metric_5 | Metric_5 Best | best epoch | other scores

**WARNING**: this command parallel computes scores for all outputs in the $output_folder, which may result in a very high usage of CPU resource.
