# vision from scratch
# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train_rp.py --task XArm-Residual-Cube-Vision-From-Scratch --enable_cameras --num_envs 1 --load_run 2025-05-05_19-02-44_vision_from_scratch_ts4_hist54_DMR --checkpoint model_1400

# student (vision): 
# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train_rp.py --task XArm-Residual-Cube-Student --enable_cameras --num_envs 1024 --headless --load_run teacher95_dmr3 --resume True --run_name distillation_from95_gl4_lr5e-4 --max_iterations 5000

# teacher (state-based):
# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train_rp.py --task XArm-Residual-Cube-Teacher --num_envs 1024 --headless --run_name state_teacher_retrain_75_epoch_dmr3



### INSERTION ###
# phase 1
# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train_rp.py --task XArm-Residual-Insertion-Teacher --num_envs 1024 --headless --run_name insertion_phase1_raw80 

# phase 2
# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train_rp.py --task XArm-Residual-Insertion-Teacher --num_envs 1024 --headless --run_name insertion_phase2_dmr --load_run 2025-05-10_02-43-06_insertion_phase1_raw80 --resume True --max_iterations 5000 

# distill
# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train_rp.py --task XArm-Residual-Insertion-Student --enable_cameras --num_envs 512 --headless --load_run teacher92_dmr --resume True --run_name distillation_insertion_v0 --max_iterations 5000


# phase 1
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train_rp.py --task XArm-Residual-Insertion-Teacher --num_envs 1 --run_name insertion_all_together_small_dmr

# phase 2
# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train_rp.py --task XArm-Residual-Insertion-Teacher --num_envs 1024 --headless --run_name insertion_phase2_dmr --load_run 2025-05-10_02-43-06_insertion_phase1_raw80 --resume True --max_iterations 5000 
