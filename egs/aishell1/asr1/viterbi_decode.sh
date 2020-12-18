#!/bin/bash
# general configuration
. ./path.sh || exit 1
train_set=train_sp
train_dev=dev

dumpdir=dump
model=$1

#python ../../../espnet/bin/asr_recog.py \
#	--ngpu 1 \
#        --viterbi true \
#        --model ${model} \
#	--recog-json dump/${train_set}/deltafalse/data.json \
#        --result-label dump/${train_set}/deltafalse/data_aligned_new.json

python ../../../espnet/bin/asr_recog.py \
        --ngpu 1 \
        --viterbi true \
        --model ${model} \
        --recog-json dump/${train_dev}/deltafalse/data.json \
        --result-label dump/${train_dev}/deltafalse/data_aligned_new.json


