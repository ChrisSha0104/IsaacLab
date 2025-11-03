CUDA_VISIBLE_DEVICES=1 python scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Factory-Xarm-PegInsert-Direct-v0  --num_envs 128 \
    --track --wandb-project-name FactoryXarm --wandb-name xarm-peginsert-fixed-ctrl-adm-v0 --wandb-entity ss7050-columbia \
    --headless