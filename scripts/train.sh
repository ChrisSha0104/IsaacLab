# state based
# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train_rp.py --task XArm-Residual-Cube-State-v2 --max_iterations 2000 --num_envs 3 --run_name test_del --headless

# cam
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train_rp.py --task XArm-Residual-Cube-v5 --enable_cameras --headless --num_envs 512 --run_name deployable_zed_no_empnorm --max_iterations 2000
