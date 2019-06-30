Implementation of the NLI model in our ACL 2019 paper: [Augmenting Neural Networks with First-order Logic](https://svivek.com/research/publications/li2019augmenting.pdf)
```
@inproceedings{li2019augmenting,
      author    = {Li, Tao and Srikumar, Vivek},
      title     = {Augmenting Neural Networks with First-order Logic},
      booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
      year      = {2019}
  }
```

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
```
python3 snli_extract.py --data ./data/snli_1.0/snli_1.0_dev.txt --output ./data/snli_1.0/dev
python3 snli_extract.py --data ./data/snli_1.0/snli_1.0_train.txt --output ./data/snli_1.0/train
python3 snli_extract.py --data ./data/snli_1.0/snli_1.0_test.txt --output ./data/snli_1.0/test

python3 preprocess.py --glove ./data/glove.840B.300d.txt --dir ./data/snli_1.0/
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
GPUID=0
CONSTR_W=n2
RHO_W=2
CONSTR_C=n3
RHO_C=1
RATIO=1
PERC=$(python -c "print(int($RATIO*100))")
SEED=1
python3 -u train.py --gpuid $GPUID --dir ./data/snli_1.0/ --train_res train.content_word.json,train.all_rel.json --val_res dev.content_word.json,dev.all_rel.json --within_constr ${CONSTR_W} --rho_w ${RHO_W} --cross_constr ${CONSTR_C} --rho_c ${RHO_C} --constr_on 1,2,3 --encoder rnn --rnn_type lstm --dropout 0.2 --epochs 100 --learning_rate 0.0001 --clip 5 --percent ${RATIO} --seed ${SEED} --save_file models/${CONSTR_W//,}_rho${RHO_W}_${CONSTR_C//,}_rho${RHO_C//.}_bilstm_lr00001_perc${PERC}_seed${SEED} | tee models/${CONSTR_W//,}_rho${RHO_W}_${CONSTR_C//,}_rho${RHO_C//.}_bilstm_lr00001_perc${PERC}_seed${SEED}.txt
```

For evaluation, remeber to change corresponding parameters in the ```eval.py```. Expect to see accuracies as reported in our paper.


## Issues & To-dos
- [ ] Add the machine comprehension model and the text chunking model.