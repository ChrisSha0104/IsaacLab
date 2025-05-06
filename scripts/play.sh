# vision from scratch
# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play_rp.py --task XArm-Residual-Cube-Vision-From-Scratch --enable_cameras --num_envs 1 --load_run 2025-05-05_19-02-44_vision_from_scratch_ts4_hist54_DMR --checkpoint model_1400

# student (vision):
./isaaclab.sh -p source/standalone/workflows/rsl_rl/play_rp.py --task XArm-Residual-Cube-Student --enable_cameras --num_envs 32 --load_run 2025-05-06_17-50-59_distillation_test

# teacher (state-based):
# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play_rp.py --task XArm-Residual-Cube-Teacher --num_envs 49 --load_run 2025-05-06_12-56-26_state_full_rollout_redu_res_rate

