CUDA_VISIBLE_DEVICES=1 python scripts/reinforcement_learning/rl_games/play.py \
    --task Isaac-Factory-Xarm-PegInsert-Direct-v0  --num_envs 1 \
    --checkpoint logs/rl_games/FactoryXarm/peg_insert_v2/nn/FactoryXarm.pth