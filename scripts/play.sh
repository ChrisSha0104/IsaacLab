# CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rl_games/play.py \
#     --task Isaac-Factory-Xarm-GearMesh-Residual  --num_envs 1 \
#     --checkpoint logs/rl_games/FactoryXarm/gear_mesh_residual_v0/nn/FactoryXarm.pth

# CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rl_games/play.py \
#     --task Isaac-Factory-Xarm-GearMesh-Residual-NoBase  --num_envs 1 \
#     --checkpoint logs/rl_games/FactoryXarm/gear_mesh_residual_no_base/nn/FactoryXarm.pth

CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rl_games/play.py \
    --task Isaac-Factory-Xarm-GearMesh-Residual-AddDelta  --num_envs 1 \
    --checkpoint logs/rl_games/FactoryXarm/gear_mesh_residual_add_delta/nn/FactoryXarm.pth