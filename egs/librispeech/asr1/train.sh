#!/bin/bash
# general configuration
. ./path.sh || exit 1

ngpu=4
debugmode=1
verbose=0
resume=

workdir=$(pwd)
config=${workdir}/conf/train_streaming_transformer.yaml

cd  ../../../

expname=streaming_transformer
expdir=exp
exppath=${expdir}/${expname}

mkdir -p ${exppath}

python espnet/bin/asr_train.py \
    --config ${config} \
    --preprocess-conf ${workdir}/conf/specaug.yaml \
    --ngpu ${ngpu} \
    --outdir ${expdir}/results \
    --train-json dump/train_960/deltafalse/data_aligned.json \
    --valid-json dump/dev/data_aligned.json \
    --resume ${resume} \
    --dict data/lang_char/train_960_unigram5000_units.txt

