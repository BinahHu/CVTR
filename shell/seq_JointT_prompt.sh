export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
LOG_NAME=2card_seq_JointT_ChineseCLIP_attnadapterv2_batch512_deepattndist_proj_svd_diff_12layers_factor2
OUT_DIR=/mnt/log/log/CTP/${LOG_NAME}

cd ../train/
python -m torch.distributed.run --nproc_per_node=8 --master_port=12600 train_JointT_prompt.py \
--config ../configs/exp/free.yaml \
--base_config ../configs/base_seqF.yaml \
--output_dir ${OUT_DIR} \
2>&1 | tee /mnt/log/log/CTP/logger/${LOG_NAME}.log
