# state based
# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train_rp.py --task XArm-Residual-Cube-State-v2 --max_iterations 2000 --num_envs 3 --run_name test_del --headless

# cam
# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train_rp.py --task XArm-Residual-Cube-v5 --enable_cameras --headless --num_envs 512 --run_name deployable_zed_no_empnorm_alpha0.1 --max_iterations 2000

# gear task
# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train_rp.py --task XArm-Residual-Gear-v0 --enable_cameras --headless --num_envs 3 --run_name gear_test --max_iterations 100


# battery task
# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train_rp.py --task XArm-Residual-Battery-v0 --enable_cameras --headless --num_envs 512 --run_name battery_cam_abs_v1 --max_iterations 2000

# battery state-based
# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train_rp.py --task XArm-Residual-Battery-v0 --enable_cameras --headless --num_envs 2048 --run_name battery_state_abs_no_noise_no_falling --max_iterations 2000














# new cube
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train_rp.py --task XArm-Residual-Cube-State-v0 --enable_cameras --headless --num_envs 2048 --run_name reduced_residual_scale_lowered_fingertip_inst_rew --max_iterations 5000
