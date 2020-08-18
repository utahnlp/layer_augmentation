<p align="center"><img width="80%" src="logo.png" /></p>

Implementation of the NLI model in our ACL 2019 paper: [Augmenting Neural Networks with First-order Logic](https://arxiv.org/abs/1906.06298)
```
@inproceedings{li2019augmenting,
      author    = {Li, Tao and Srikumar, Vivek},
      title     = {Augmenting Neural Networks with First-order Logic},
      booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
      year      = {2019}
  }
```

**For the QA model, check out [here](https://github.com/utahnlp/layer_augmentation_qa).**

## Prerequisites
```
pytorch 0.4.1
numpy
h5py
spacy 2.0.11 (with en model)
glove.840B.300d.txt (under ./data/)
```
Besides above, make sure snli 1.0 data is unpacked to ```./data/snli_1.0/```, e.g. ```./data/snli_1.0/snli_1.0_dev.txt```.

Also unzip the file ```./data/snli_1.0/conceptnet_rel.zip``` and put all files directly under path ```./data/snli_1.0/```.

## 0. Preprocessing

First run extraction code:
```
python3 snli_extract.py --data ./data/snli_1.0/snli_1.0_dev.txt --output ./data/snli_1.0/dev
python3 snli_extract.py --data ./data/snli_1.0/snli_1.0_train.txt --output ./data/snli_1.0/train
python3 snli_extract.py --data ./data/snli_1.0/snli_1.0_test.txt --output ./data/snli_1.0/test
```
Alternatively, you can unzip the ``snli_extracted.zip`` file into ``./data/snli_1.0/`` directory. This is recommended for reproduction.

Then run batching code:
```
python3 preprocess.py --glove ./data/glove.840B.300d.txt --dir ./data/snli_1.0/ --batch_size 48
python3 get_pretrain_vecs.py --glove ./data/glove.840B.300d.txt --dict ./data/snli_1.0/snli.word.dict --output ./data/snli_1.0/glove
python3 get_char_idx.py --dict ./data/snli_1.0/snli.allword.dict --token_l 16 --freq 5 --output ./data/snli_1.0/char
```

## 1. Training
```
mkdir ./models

python3 -u train.py --gpuid [GPUID] --dir ./data/snli_1.0/ --train_data snli-train.hdf5 --val_data snli-val.hdf5 --word_vecs glove.hdf5 \
--encoder rnn --rnn_type lstm  --attention local --classifier local --dropout 0.2 --epochs 100 --learning_rate 0.0001 --clip 5 \
--save_file models/lstm_clip5_adam_lr00001 | tee models/lstm_clip5_adam_lr00001.txt
```
Expect to see dev accuracy around ```87```.

## 2. Evaluation
First redo evaluation on the dev set to make sure we can get exactly the same F1 as reported during training:
```
python3 -u eval.py --gpuid [GPUID] --dir ./data/snli_1.0/ --data snli-test.hdf5 --word_vecs glove.hdf5 \
--encoder rnn --rnn_type lstm --attention local --classifier local --dropout 0.0 \
--load_file ./models/lstm_clip5_adam_lr00001 | tee models/lstm_clip5_adam_lr00001.evallog.txt
```
Expect to see test accuracy to be around ```87```.


## 3. Augmented Models
To train augmented models using the constraints ```N1```, ```N2```, and ```N3``` in our paper, simply run:
```
GPUID=[GPUID]
CONSTR_W=n2
RHO_W=2
CONSTR_C=n3
RHO_C=1
RATIO=1
PERC=$(python -c "print(int($RATIO*100))")
SEED=1
python3 -u train.py --gpuid $GPUID --dir ./data/snli_1.0/ --train_res train.content_word.json,train.all_rel.json \
--val_res dev.content_word.json,dev.all_rel.json \
--within_constr ${CONSTR_W} --rho_w ${RHO_W} --cross_constr ${CONSTR_C} --rho_c ${RHO_C} --constr_on 1,2,3 \
--encoder rnn --rnn_type lstm --dropout 0.2 --epochs 100 --learning_rate 0.0001 --clip 5 \
--percent ${RATIO} --seed ${SEED} \
--save_file models/${CONSTR_W//,}_rho${RHO_W}_${CONSTR_C//,}_rho${RHO_C//.}_bilstm_lr00001_perc${PERC}_seed${SEED} | tee models/${CONSTR_W//,}_rho${RHO_W}_${CONSTR_C//,}_rho${RHO_C//.}_bilstm_lr00001_perc${PERC}_seed${SEED}.txt
```

For evaluation, remeber to change corresponding parameters in the ```eval.py```. Expect to see accuracies as reported in our paper.


## ConceptNet

Before proceeding, please make sure you have a local instance of ConceptNet running. An example of setup can be found [here](https://www.cs.utah.edu/~tli/posts/2018/09/blog-post-3/)

For extracting edges from ConceptNet, you can refer to the following code:
```
DATASET=train
python3 -u conceptnet.py --sent1_lemma ./data/nli_aug/${DATASET}.sent1_lemma.txt --sent2_lemma ./data/nli_aug/${DATASET}.sent2_lemma.txt --worker 4 --rel syn --output ./data/nli_aug/conceptnet.syn.txt --continu 1
python3 -u conceptnet.py --sent1_lemma ./data/nli_aug/${DATASET}.sent1_lemma.txt --sent2_lemma ./data/nli_aug/${DATASET}.sent2_lemma.txt --worker 4 --rel distinct --output ./data/nli_aug/conceptnet.distinct.txt --continu 1
python3 -u conceptnet.py --sent1_lemma ./data/nli_aug/${DATASET}.sent1_lemma.txt --sent2_lemma ./data/nli_aug/${DATASET}.sent2_lemma.txt --worker 4 --rel related --output ./data/nli_aug/conceptnet.related.txt --continu 1
python3 -u conceptnet.py --sent1_lemma ./data/nli_aug/${DATASET}.sent1_lemma.txt --sent2_lemma ./data/nli_aug/${DATASET}.sent2_lemma.txt --worker 4 --rel isa --output ./data/nli_aug/conceptnet.isa.txt --continu 1

python3 constraint_preprocess.py --dir ./data/nli_aug/ --src dev.sent1_lemma.txt --targ dev.sent2_lemma.txt --output_rel all_rel --output dev
python3 constraint_preprocess.py --dir ./data/nli_aug/ --src train.sent1_lemma.txt --targ train.sent2_lemma.txt --output_rel all_rel --output train
python3 constraint_preprocess.py --dir ./data/nli_aug/ --src test.sent1_lemma.txt --targ test.sent2_lemma.txt --output_rel all_rel --output test

python3 constraint_preprocess.py --dir ./data/nli_aug/ --src dev.sent1_lemma.txt --targ dev.sent2_lemma.txt --src_pos dev.sent1_pos.txt --targ_pos dev.sent2_pos.txt --output_rel content_word --output dev
python3 constraint_preprocess.py --dir ./data/nli_aug/ --src train.sent1_lemma.txt --targ train.sent2_lemma.txt --src_pos train.sent1_pos.txt --targ_pos train.sent2_pos.txt --output_rel content_word --output train
python3 constraint_preprocess.py --dir ./data/nli_aug/ --src test.sent1_lemma.txt --targ test.sent2_lemma.txt --src_pos test.sent1_pos.txt --targ_pos test.sent2_pos.txt --output_rel content_word --output test

python3 constraint_preprocess.py --dir ./data/nli_aug/ --src dev.sent1_lemma.txt --targ dev.sent2_lemma.txt --output_rel excl_ant --output dev
python3 constraint_preprocess.py --dir ./data/nli_aug/ --src train.sent1_lemma.txt --targ train.sent2_lemma.txt --output_rel excl_ant --output train
python3 constraint_preprocess.py --dir ./data/nli_aug/ --src test.sent1_lemma.txt --targ test.sent2_lemma.txt --output_rel excl_ant --output test
```


## Issues & To-dos
- [x] Add the machine comprehension model.
- [ ] Add the text chunking model.
