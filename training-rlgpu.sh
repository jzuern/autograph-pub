export WANDB_API_KEY=8eca0f0d4e3d49c3728c9aa0e00b316c2d80012f
export NUM_GPUS=2
export DATASET_NAME=pittsburgh-2604


# TrackletNet gt-supervised
WANDB_API_KEY=$WANDB_API_KEY ~/zuern/geometric/bin/python train_regressor_pos_query.py \
  --config cfg-rlgpu.yaml \
  --target full \
  --num_gpus $NUM_GPUS \
  --dataset_name $DATASET_NAME/lanegraph -d

# TrackletNet tracklet-supervised
WANDB_API_KEY=$WANDB_API_KEY ~/zuern/geometric/bin/python train_regressor_pos_query.py \
  --config cfg-rlgpu.yaml \
  --target full \
  --num_gpus $NUM_GPUS \
  --dataset_name $DATASET_NAME/tracklets_joint -d





## SuccNet gt-supervised (TrackletNet A + D)
#WANDB_API_KEY=$WANDB_API_KEY ~/zuern/geometric/bin/python train_regressor_pos_query.py \
#  --config cfg-rlgpu.yaml \
#  --target successor \
#  --num_gpus $NUM_GPUS \
#  --input_layers rgb+drivable+angles \
#  --full-checkpoint /home/buechner/zuern/self-supervised-graph/checkpoints/XXXX.pth \
#  --dataset_name $DATASET_NAME/lanegraph


## SuccNet tracklets_raw supervised (no TrackletNet)
#WANDB_API_KEY=$WANDB_API_KEY ~/zuern/geometric/bin/python train_regressor_pos_query.py \
#  --config cfg-rlgpu.yaml \
#  --num_gpus $NUM_GPUS \
#  --target successor \
#  --input_layers rgb \
#  --dataset_name test-austin/tracklets_raw
#
## SuccNet tracklets_joint supervised (no TrackletNet)
#WANDB_API_KEY=$WANDB_API_KEY ~/zuern/geometric/bin/python train_regressor_pos_query.py \
#  --config cfg-rlgpu.yaml \
#  --num_gpus $NUM_GPUS \
#  --target successor \
#  --input_layers rgb \
#  --dataset_name $DATASET_NAME/tracklets_joint
#
#
## SuccNet tracklet-supervised (TrackletNet D)
#WANDB_API_KEY=$WANDB_API_KEY ~/zuern/geometric/bin/python train_regressor_pos_query.py \
#  --config cfg-rlgpu.yaml \
#  --num_gpus $NUM_GPUS \
#  --target successor \
#  --input_layers rgb+drivable \
#  --full-checkpoint /home/buechner/zuern/self-supervised-graph/checkpoints/XXXX.pth \
#  --dataset_name $DATASET_NAME/tracklets_joint
#
#
## SuccNet tracklet-supervised (TrackletNet D + A)
#WANDB_API_KEY=$WANDB_API_KEY ~/zuern/geometric/bin/python train_regressor_pos_query.py \
#  --config cfg-rlgpu.yaml \
#  --num_gpus $NUM_GPUS \
#  --target successor \
#  --input_layers rgb+drivable+angles \
#  --full-checkpoint /home/buechner/zuern/self-supervised-graph/checkpoints/XXXX.pth \
#  --dataset_name $DATASET_NAME/tracklets_joint
