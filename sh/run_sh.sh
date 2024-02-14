#!/bin/bash
seed=101
dataset_name="liu"
task="5w1s"
output="first"
gpu=2
kshot=1
numf=0
dataset_num=("01" "02" "03" "04" "05")

for k in "${dataset_num[@]}";do
  python ../code/main2.py --config ./liu_1shot.json \
  --seed $seed \
  --gpu $gpu \
  --kshot $kshot \
  --output $output \
  --numFreeze $numf \
  --dataset_num $k \
  --dataset_name $dataset_name \
  --task $task
done

