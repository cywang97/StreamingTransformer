#!/bin/bash
. ./path.sh || exit 1
. ./cmd.sh || exit 1
dataset='dev test'
config=conf/decode.yaml
ngpu=$1
jobperGPU=$2
((nblock=$ngpu*$jobperGPU))
startblock=1
endblock=${nblock}+1
model_dir=exp/transformer_12layer_specaug
rnnlm=exp/train_rnnlm_pytorch_lm

#average_checkpoints.py \
#    --backend pytorch \
#    --snapshots ${model_dir}/results/snapshot.ep.* \
#    --out ${model_dir}/results/model.last5.avg.best \
#    --num 5

function run(){
part=$1
cgpu=$[$2+0]
recog_set=$3
echo $cgpu
export CUDA_VISIBLE_DEVICES=$cgpu
mkdir -p ${model_dir}/decode/${recog_set}/log
${decode_cmd} ${model_dir}/decode/${recog_set}/log/decode.${part}.log \
    asr_recog.py \
    --config ${config} \
    --ngpu 1 \
    --rnnlm ${rnnlm} \
    --recog-json dump/${recog_set}/deltafalse/split${nblock}utt/data_unigram5000.${part}.json \
    --model ${model_dir}/results/model.last5.avg.best \
    --result-label ${model_dir}/decode/${recog_set}/data.${part}.json
}

function recog(){
recog_set=$1
decode_dir=${model_dir}/decode/${recog_set}

splitjson.py --parts ${nblock} dump/${recog_set}/deltafalse/data.json 

for ((i=$startblock;i<$endblock;i+=${ngpu}));
do
    for ((j=0;j<${ngpu};j++));
    do
        run $(($i+$j)) $j $recog_set &
    done

    if [ $(( $[$i+1]%${jobperGPU} )) -eq 0 ]
    then
    wait;
    fi
done
wait
score_sclite.sh ${model_dir}/decode/${recog_set} data/lang_char/train_sp_units.txt
}

for recog_set in ${dataset}; do
    recog $recog_set
done
