### CUBE ###

# old vision
# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play_rp.py --task XArm-Residual-Cube-v5 --enable_cameras --num_envs 1 --load_run 2025-04-18_16-43-13_deployable_zed --checkpoint model_700

# new vision
# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play_rp.py --task XArm-Residual-Cube-v5 --enable_cameras --num_envs 1 --load_run 2025-04-23_16-25-31_deployable_zed_no_empnorm_alpha0.1 --checkpoint model_700

# play state based policy
# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play_rp.py --task XArm-Residual-Cube-State-v2 --num_envs 1 --load_run 2025-04-16_12-13-45_deployable_state_0.3s_hist --checkpoint model_1000

# current policy in real
# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play_rp.py --task XArm-Residual-Cube-State-v2 --num_envs 1 --load_run deployable_1.5s_hist #--checkpoint model_1000



### GEAR ###
# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play_rp.py --task XArm-Residual-Battery-v0 --enable_cameras --num_envs 1 --load_run 2025-04-28_15-19-11_battery_state_abs_no_noise_no_falling



###NEW CUBE###

# STATEBASED!!! 
# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play_rp.py --task XArm-Residual-Cube-State-v0 --num_envs 16 --load_run 5-5_STATE_asym_hist54_early_term_DMR


# VISION!!!
./isaaclab.sh -p source/standalone/workflows/rsl_rl/play_rp.py --task XArm-Residual-Cube-Vision-v0 --enable_cameras --num_envs 1 --load_run 2025-05-05_19-02-44_vision_from_scratch_ts4_hist54_DMR --checkpoint model_1400