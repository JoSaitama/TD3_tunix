#!/usr/bin/env bash
set -e

source ~/td3-venv/bin/activate
cd ~/TD3_tunix

python -m td3_tunix.training.train_td3 \
  --env_name halfcheetah \
  --seed 0 \
  --total_steps 20000 \
  --start_timesteps 2000 \
  --eval_freq 5000 \
  --replay_size 100000 \
  --batch_size 256 \
  --max_episode_steps 1000 \
  --num_eval_episodes 3
