#!/bin/bash

# デフォルト値
sleep_time=0

# 引数解析
while [[ "$1" == --* ]]; do
    case "$1" in
        --sleep=*)
            sleep_time="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 必要な引数チェック
if [ $# -lt 5 ]; then
    echo "Usage: ./train_allegro_kuka.sh <task> <experiment_prefix> <num_policies> <num_envs> <wandb_tags> <extra_args>"
    exit 1
fi

# 必要な変数の設定
gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
(( exp_per_gpu=($3+$gpus-1)/$gpus ))

task=$1
experiment_prefix=$2
num_policies=$3
num_envs=$4
tags=$5
extra_args=${@:6}

# スリープタイム
echo "Sleeping for $sleep_time seconds..."
sleep $sleep_time

# 実行
hostname

python3 -m isaacgymenvs.pbt.launcher.run \
    --experiments_per_gpu=$exp_per_gpu \
    --num-policies=$num_policies \
    --num-envs=$num_envs \
    --task=$task \
    --num_gpus=$gpus \
    --wandb-tags=$tags \
    --experiment_prefix=$experiment_prefix \
    $extra_args