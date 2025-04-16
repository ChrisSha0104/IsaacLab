# state based
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train_rp.py --task XArm-Residual-Cube-State-v2 --max_iterations 2000 --num_envs 2048 --run_name deployable_state_no_scaling --headless

# cam
# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train_rp.py --task XArm-Residual-Cube-v4 --enable_cameras --headless --num_envs 512 --run_name deployable_cam_reduced_obs_cln_depth_histlen_10 --max_iterations 2000
