export WANDB_API_KEY=8eca0f0d4e3d49c3728c9aa0e00b316c2d80012f
export DATASET_NAME=all-3004

## TrackletNet gt-supervised
#CUDA_VISIBLE_DEVICES=0,1 WANDB_API_KEY=$WANDB_API_KEY ~/zuern/geometric/bin/python train_regressor_pos_query.py \
#  --config cfg-rlgpu.yaml \
#  --target full \
#  --dataset_name $DATASET_NAME/lanegraph
#
#sleep 1
#
## TrackletNet tracklet-supervised
#CUDA_VISIBLE_DEVICES=2,3 WANDB_API_KEY=$WANDB_API_KEY ~/zuern/geometric/bin/python train_regressor_pos_query.py \
#  --config cfg-rlgpu.yaml \
#  --target full \
#  --dataset_name $DATASET_NAME/tracklets_raw
#
#

# SuccNet tracklets_raw supervised (no TrackletNet)
CUDA_VISIBLE_DEVICES=0 =$WANDB_API_KEY ~/zuern/geometric/bin/python train_regressor_pos_query.py \
  --config cfg-rlgpu.yaml \
  --target successor \
  --input_layers rgb \
  --dataset_name $DATASET_NAME/tracklets_raw &

sleep 1

# SuccNet tracklets_joint supervised (no TrackletNet)
CUDA_VISIBLE_DEVICES=1 WANDB_API_KEY=$WANDB_API_KEY ~/zuern/geometric/bin/python train_regressor_pos_query.py \
  --config cfg-rlgpu.yaml \
  --target successor \
  --input_layers rgb \
  --dataset_name $DATASET_NAME/tracklets_joint &

sleep 1

# SuccNet tracklets_joint supervised (TrackletNet D)
CUDA_VISIBLE_DEVICES=2 WANDB_API_KEY=$WANDB_API_KEY ~/zuern/geometric/bin/python train_regressor_pos_query.py \
  --config cfg-rlgpu.yaml \
  --target successor \
  --input_layers rgb+drivable \
  --full-checkpoint checkpoints/civilized-bothan-187/e-150.pth \
  --dataset_name $DATASET_NAME/tracklets_joint &

sleep 1

# SuccNet tracklets_joint supervised (TrackletNet D + A)
CUDA_VISIBLE_DEVICES=3 WANDB_API_KEY=$WANDB_API_KEY ~/zuern/geometric/bin/python train_regressor_pos_query.py \
  --config cfg-rlgpu.yaml \
  --target successor \
  --input_layers rgb+drivable+angles \
  --full-checkpoint checkpoints/civilized-bothan-187/e-150.pth \
  --dataset_name $DATASET_NAME/tracklets_joint



### SuccNet gt-supervised
#CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=$WANDB_API_KEY ~/zuern/geometric/bin/python train_regressor_pos_query.py \
#  --config cfg-rlgpu.yaml \
#  --target successor \
#  --input_layers rgb \
#  --dataset_name $DATASET_NAME/lanegraph

### SuccNet gt-supervised (TrackletNet A + D)
#CUDA_VISIBLE_DEVICES=1 WANDB_API_KEY=$WANDB_API_KEY ~/zuern/geometric/bin/python train_regressor_pos_query.py \
#  --config cfg-rlgpu.yaml \
#  --target successor \
#  --input_layers rgb+drivable \
#  --full-checkpoint /home/buechner/zuern/self-supervised-graph/checkpoints/XXXX.pth \
#  --dataset_name $DATASET_NAME/lanegraph



### SuccNet gt-supervised (TrackletNet A + D)
#CUDA_VISIBLE_DEVICES=2 WANDB_API_KEY=$WANDB_API_KEY ~/zuern/geometric/bin/python train_regressor_pos_query.py \
#  --config cfg-rlgpu.yaml \
#  --target successor \
#  --input_layers rgb+drivable+angles \
#  --full-checkpoint /home/buechner/zuern/self-supervised-graph/checkpoints/XXXX.pth \
#  --dataset_name $DATASET_NAME/lanegraph

