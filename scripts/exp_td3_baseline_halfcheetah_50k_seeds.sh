#!/usr/bin/env bash
set -e

source ~/td3-venv/bin/activate
cd ~/TD3_tunix

mkdir -p logs/baseline_50k

for seed in 0 1 2 3 4
do
  echo "===== Running TD3 baseline HalfCheetah seed ${seed} ====="

  python -u -m td3_tunix.training.train_td3 \
    --env_name halfcheetah \
    --seed ${seed} \
    --total_steps 50000 \
    --start_timesteps 5000 \
    --eval_freq 10000 \
    --replay_size 100000 \
    --batch_size 256 \
    --max_episode_steps 1000 \
    --num_eval_episodes 5 \
    2>&1 | tee logs/baseline_50k/td3_baseline_halfcheetah_50k_seed${seed}.log
done
