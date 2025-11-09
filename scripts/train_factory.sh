# CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rl_games/train.py \
#     --task Isaac-Factory-Xarm-PegInsert-Direct-v0  --num_envs 128 \
#     --track --wandb-project-name FactoryXarm --wandb-name xarm-peginsert-fixed-ctrl-adm-alpha1.0 --wandb-entity ss7050-columbia \
#     --headless

CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Factory-Xarm-GearMesh-Residual  --num_envs 128 \
    --track --wandb-project-name FactoryXarm --wandb-name xarm-gearmesh-residual-v1 --wandb-entity ss7050-columbia \
    --headless