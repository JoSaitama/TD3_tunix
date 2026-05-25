#!/usr/bin/env bash
set -e

source ~/td3-venv/bin/activate
cd ~/TD3_tunix

mkdir -p logs/baseline_100k

for seed in 0 1 2
do
  echo "===== Running TD3 baseline HalfCheetah 100k seed ${seed} ====="

  python -u -m td3_tunix.training.train_td3 \
    --env_name halfcheetah \
    --seed ${seed} \
    --total_steps 100000 \
    --start_timesteps 10000 \
    --eval_freq 5000 \
    --replay_size 200000 \
    --batch_size 256 \
    --max_episode_steps 1000 \
    --num_eval_episodes 3 \
    2>&1 | tee logs/baseline_100k/td3_baseline_halfcheetah_100k_seed${seed}.log
done
