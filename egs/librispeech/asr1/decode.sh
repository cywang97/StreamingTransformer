#!/bin/bash
dataset='test_clean test_other'
config=conf/decode.yaml
ngpu=$1
jobperGPU=$2
((nblock=$ngpu*$jobperGPU))
startblock=0
endblock=${nblock}
model_dir=exp/streaming_transformer
rnnlm=exp/irielm.ep11.last5.avg/rnnlm.model.best

average_checkpoints.py \
    --backend pytorch \
    --snapshots ${model_dir}/results/snapshot.ep.* \
    --out ${model_dir}/results/model.last5.avg.best \
    --num 5

function run(){
part=$1
cgpu=$[$2+0]
recog_set=$3
echo $cgpu
export CUDA_VISIBLE_DEVICES=$cgpu
python ../../../espnet/bin/asr_recog.py \
    --config ${config} \
    --ngpu 1 \
    --rnnlm ${rnnlm} \
    --recog-json dump/${recog_set}/data_aligned_${part}.json \
    --model ${model_dir}/results/model.last5.avg.best \
    --result-label ${model_dir}/decode/${recog_set}/data.${part}.json
}

function recog(){
recog_set=$1
decode_dir=${model_dir}/decode/${recog_set}
mkdir -p ${decode_dir}/data
python  split_data.py \
    --parts ${nblock} \
    --json dump/${recog_set}/data_unigram5000.json \
    --datadir ${decode_dir}/data

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

    echo "decode $[$i+$ngpu]/${nblock} done"
done
#wait
echo "decode all done!"

python merge_data.py \
    --parts ${nblock} \
    --result-dir ${decode_dir} \
    --result-label ${decode_dir}/data.json
echo "merge result json done!"
}

for recog_set in ${dataset}; do
    recog $recog_set
done
