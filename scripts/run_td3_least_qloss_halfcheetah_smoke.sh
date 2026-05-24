#!/usr/bin/env bash
set -e

source ~/td3-venv/bin/activate
cd ~/TD3_tunix

python -m td3_tunix.training.train_td3_least_qloss \
  --env_name halfcheetah \
  --seed 0 \
  --total_steps 5000 \
  --start_timesteps 1000 \
  --eval_freq 1000 \
  --replay_size 50000 \
  --batch_size 256 \
  --max_episode_steps 1000 \
  --num_eval_episodes 3 \
  --least_start_steps 2000 \
  --least_buffer_size 20 \
  --least_min_ref_episodes 2 \
  --least_min_episode_steps 50 \
  --omega_clip_min 0.2 \
  --omega_clip_max 5.0
