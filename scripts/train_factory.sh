# CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rl_games/train.py \
#     --task Isaac-Factory-Xarm-PegInsert-Direct-v0  --num_envs 128 \
#     --track --wandb-project-name FactoryXarm --wandb-name xarm-peginsert-fixed-ctrl-adm-alpha1.0 --wandb-entity ss7050-columbia \
#     --headless

# CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rl_games/train.py \
#     --task Isaac-Factory-Xarm-GearMesh-Residual  --num_envs 128 \
#     --track --wandb-project-name FactoryXarm --wandb-name xarm-gearmesh-residual-async-v2 --wandb-entity ss7050-columbia \
#     --headless agent.params.config.full_experiment_name=gear_mesh_residual_async_v2

# CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rl_games/train.py \
#     --task Isaac-Factory-Xarm-GearMesh-Residual-NoBase  --num_envs 128 \
#     --track --wandb-project-name FactoryXarm --wandb-name xarm-gearmesh-residual-no-base-v1 --wandb-entity ss7050-columbia \
#     --headless agent.params.config.full_experiment_name=gear_mesh_residual_no_base_v1

# CUDA_VISIBLE_DEVICES=1 python scripts/reinforcement_learning/rl_games/train.py \
#     --task Isaac-Factory-Xarm-GearMesh-Residual-AddDelta  --num_envs 128 \
#     --track --wandb-project-name FactoryXarm --wandb-name xarm-gearmesh-residual-add-delta-v1 --wandb-entity ss7050-columbia \
#     --headless agent.params.config.full_experiment_name=gear_mesh_residual_add_delta_v1

# CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rl_games/train.py \
#     --task Isaac-Factory-Xarm-PegInsert-Residual-AddDelta  --num_envs 128 \
#     --track --wandb-project-name FactoryXarm --wandb-name xarm-peginsert-residual-add-delta-fixed-rew-v3 --wandb-entity ss7050-columbia \
#     --headless agent.params.config.full_experiment_name=peg_insert_residual_add_delta_fixed_rew_v3

# CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rl_games/train.py \
#     --task Isaac-Factory-Xarm-GearMesh-Residual-Sparse  --num_envs 128 \
#     --track --wandb-project-name FactoryXarm --wandb-name xarm-gearmesh-residual-sparse-v0 --wandb-entity ss7050-columbia \
#     --headless agent.params.config.full_experiment_name=gear_mesh_residual_sparse_v0


# CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rl_games/train.py \
#     --task Isaac-Factory-Xarm-PegInsert-Residual-Sparse  --num_envs 128 \
#     --track --wandb-project-name FactoryXarm --wandb-name xarm-peginsert-residual-sparse-h60-firm-v6 --wandb-entity ss7050-columbia \
#     --headless agent.params.config.full_experiment_name=peg_insert_residual_sparse_h60_firm_v6

time="$(date +%Y-%m-%d_%H-%M-%S)"

CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Factory-Xarm-GearMesh-Residual-Sparse-New  --num_envs 128 \
    --track --wandb-project-name FactoryXarm --wandb-name xarm-gearmesh-residual-dmr-v0 --wandb-entity ss7050-columbia \
    --headless agent.params.config.full_experiment_name=${time}_gear_mesh_residual_sparse_dmr_v0