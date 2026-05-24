#!/usr/bin/env bash
set -e

source ~/td3-venv/bin/activate
cd ~/TD3_tunix

python -m td3_tunix.training.train_td3_least_qloss \
  --env_name halfcheetah \
  --seed 0 \
  --total_steps 20000 \
  --start_timesteps 2000 \
  --eval_freq 5000 \
  --replay_size 100000 \
  --batch_size 256 \
  --max_episode_steps 1000 \
  --num_eval_episodes 3 \
  --least_start_steps 10000 \
  --least_buffer_size 20 \
  --least_min_ref_episodes 5 \
  --least_min_episode_steps 100 \
  --omega_clip_min 0.1 \
  --omega_clip_max 10.0 \
  --dynamic_buffer \
  --dynamic_update_freq 500 \
  --dynamic_entropy_bins 20 \
  --dynamic_entropy_gamma 0.2 \
  --dynamic_adjust_scale 2 \
  --dynamic_min_active_size 5 \
  --dynamic_initial_active_size 20 \
  --adaptive_noise \
  --noise_window_episodes 20 \
  --noise_target_stop_rate 0.35 \
  --noise_max 0.35 \
  --noise_smoothing 0.25 \
  --noise_min_events 5
