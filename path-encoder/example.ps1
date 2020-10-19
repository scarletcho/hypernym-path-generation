python main.py --train data/split/wn18rr_adjust/neg/train_verb_path_18rr_12r.r --dev data/split/wn18rr_adjust/neg/valid_verb_path_18rr_12r.r --query data/verb_wn18_path --neg --gamma 0.5 --lr 0.001 -e 80 --valid --cuda -s --hidden 1024
python main.py --test data/split/wn18rr_adjust/neg/test_verb_path_18rr_12r.r --query data/verb_wn18_path --neg --gamma 0.1 --lr 0.001 -e 75 -o --seq_len 20 -l checkpoint/train_verb_path_18rr_12r.r_dr_0_lr_0.001_gamma_0.1_hid_1024/epoch75.chkpt --cuda
python ./get_output.py ./test_output/ 10 test
