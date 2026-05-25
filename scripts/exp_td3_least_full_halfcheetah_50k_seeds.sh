#!/usr/bin/env bash
set -e

source ~/td3-venv/bin/activate
cd ~/TD3_tunix

mkdir -p logs/least_full_50k

for seed in 0 1 2 3 4
do
  echo "===== Running TD3 + full LEAST HalfCheetah seed ${seed} ====="

  python -u -m td3_tunix.training.train_td3_least_qloss \
    --env_name halfcheetah \
    --seed ${seed} \
    --total_steps 50000 \
    --start_timesteps 5000 \
    --eval_freq 10000 \
    --replay_size 100000 \
    --batch_size 256 \
    --max_episode_steps 1000 \
    --num_eval_episodes 5 \
    --least_start_steps 20000 \
    --least_buffer_size 50 \
    --least_min_ref_episodes 10 \
    --least_min_episode_steps 200 \
    --omega_clip_min 0.1 \
    --omega_clip_max 10.0 \
    --dynamic_buffer \
    --dynamic_update_freq 2000 \
    --dynamic_entropy_bins 20 \
    --dynamic_entropy_gamma 0.5 \
    --dynamic_adjust_scale 1 \
    --dynamic_min_active_size 15 \
    --dynamic_initial_active_size 30 \
    --adaptive_noise \
    --noise_window_episodes 30 \
    --noise_target_stop_rate 0.65 \
    --noise_max 0.25 \
    --noise_smoothing 0.15 \
    --noise_min_events 10 \
    2>&1 | tee logs/least_full_50k/td3_least_full_halfcheetah_50k_seed${seed}.log
done
