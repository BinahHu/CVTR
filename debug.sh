export PYTHONPATH="${PYTHONPATH}:/mnt/code/CVTR/"


set -e
HOST_NUM=1
INDEX=0
CHIEF_IP=localhost
HOST_GPU_NUM=2

#python3 -m torch.distributed.launch  --nnodes=$HOST_NUM  --node_rank=$INDEX  --master_addr $CHIEF_IP  --nproc_per_node $HOST_GPU_NUM  --master_port 8081  run/train_egoclip.py --config configs/pt/egoclip.json
python3 -m torch.distributed.launch  --nnodes=$HOST_NUM  --master_addr $CHIEF_IP  --nproc_per_node $HOST_GPU_NUM  --master_port 8081 dataset/EgoClipDataset.py