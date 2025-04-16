# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play_rp.py --task XArm-Residual-Cube-v4 --enable_cameras --num_envs 2 --load_run 2025-04-10_21-46-28_deployable_cam_reduced_obs_cln_depth_histlen_10

# play state based policy
./isaaclab.sh -p source/standalone/workflows/rsl_rl/play_rp.py --task XArm-Residual-Cube-State-v2 --num_envs 1 --load_run state-based-deployable-model-1500 --checkpoint model_1500
