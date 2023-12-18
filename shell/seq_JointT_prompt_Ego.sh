export CUDA_VISIBLE_DEVICES=0,1,2,3
LOG_NAME=EgoIT_seq_JointT_CLIP_finetune_batch256_scratch
OUT_DIR=/mnt/log/log/CTP/${LOG_NAME}

cd ../train/
python -m torch.distributed.run --nproc_per_node=4 --master_port=12600 train_JointT_prompt.py \
--config ../configs/exp/free.yaml \
--base_config ../configs/base_seqF_EgoIT.yaml \
--output_dir ${OUT_DIR} \
2>&1 | tee /mnt/log/log/CTP/logger/${LOG_NAME}.log
