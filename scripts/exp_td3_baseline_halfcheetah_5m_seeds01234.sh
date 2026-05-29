#!/usr/bin/env bash
set -e

source ~/td3-venv/bin/activate
cd ~/TD3_tunix

mkdir -p logs/baseline_5m

for seed in 0 1 2 3 4
do
  echo "===== Running TD3 baseline HalfCheetah 5M seed ${seed} ====="
  echo "Start time: $(date)"

  python -u -m td3_tunix.training.train_td3 \
    --env_name halfcheetah \
    --seed ${seed} \
    --total_steps 5000000 \
    --start_timesteps 25000 \
    --eval_freq 5000 \
    --replay_size 1000000 \
    --batch_size 100 \
    --max_episode_steps 1000 \
    --num_eval_episodes 10 \
    2>&1 | tee logs/baseline_5m/td3_baseline_halfcheetah_5m_seed${seed}.log

  echo "Finished seed ${seed} at: $(date)"
done
