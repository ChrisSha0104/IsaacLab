# vision from scratch
# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play_rp.py --task XArm-Residual-Cube-Vision-From-Scratch --enable_cameras --num_envs 1 --load_run 2025-05-05_19-02-44_vision_from_scratch_ts4_hist54_DMR --checkpoint model_1400

# student (vision):
# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play_rp.py --task XArm-Residual-Cube-Student --enable_cameras --num_envs 100 --headless --load_run 2025-05-07_12-01-09_distillation_from95

# teacher (state-based):
./isaaclab.sh -p source/standalone/workflows/rsl_rl/play_rp.py --task XArm-Residual-Cube-Teacher --enable_cameras --num_envs 1 --load_run 2025-05-07_02-27-43_state_teacher_retrain_75_epoch_dmr3

