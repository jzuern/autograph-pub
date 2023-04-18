export WANDB_API_KEY=8eca0f0d4e3d49c3728c9aa0e00b316c2d80012f
export NUM_GPUS=1


### TrackletNet gt-supervised
#WANDB_API_KEY=$WANDB_API_KEY ~/anaconda3/envs/geometric/bin/python train_regressor_pos_query.py \
#  --config cfg-rittersport.yaml \
#  --num_gpus $NUM_GPUS \
#  --target full \
#  --dataset_name test-austin/lanegraph -d

### TrackletNet tracklet-supervised
#WANDB_API_KEY=$WANDB_API_KEY ~/anaconda3/envs/geometric/bin/python train_regressor_pos_query.py \
#  --config cfg-rittersport.yaml \
#  --num_gpus $NUM_GPUS \
#  --target full \
#  --dataset_name test-austin/tracklets_joint -d

## SuccNet gt-supervised A + D
#WANDB_API_KEY=$WANDB_API_KEY ~/anaconda3/envs/geometric/bin/python train_regressor_pos_query.py \
#  --config cfg-rittersport.yaml \
#  --num_gpus $NUM_GPUS \
#  --target successor \
#  --input_layers rgb+drivable+angles \
#  --full-checkpoint /data/autograph/checkpoints/local_run_full/e-003.pth \
#  --dataset_name test-austin/lanegraph -d

#
## SuccNet tracklets_raw supervised (no TrackletNet)
#WANDB_API_KEY=$WANDB_API_KEY ~/anaconda3/envs/geometric/bin/python train_regressor_pos_query.py \
#  --config cfg-rittersport.yaml \
#  --num_gpus $NUM_GPUS \
#  --target successor \
#  --input_layers rgb \
#  --full-checkpoint /data/autograph/checkpoints/local_run_full/e-003.pth \
#  --dataset_name test-austin/tracklets_raw -d

## SuccNet tracklets_joint supervised (no TrackletNet)
#WANDB_API_KEY=$WANDB_API_KEY ~/anaconda3/envs/geometric/bin/python train_regressor_pos_query.py \
#  --config cfg-rittersport.yaml \
#  --num_gpus $NUM_GPUS \
#  --target successor \
#  --input_layers rgb \
#  --full-checkpoint /data/autograph/checkpoints/local_run_full/e-003.pth \
#  --dataset_name test-austin/tracklets_joint -d

#
## SuccNet tracklet-supervised (TrackletNet D)
#WANDB_API_KEY=$WANDB_API_KEY ~/anaconda3/envs/geometric/bin/python train_regressor_pos_query.py \
#  --config cfg-rittersport.yaml \
#  --target successor \
#  --input_layers rgb+drivable \
#  --full-checkpoint /data/autograph/checkpoints/local_run_full/e-003.pth \
#  --dataset_name test-austin/tracklets_joint -d


## SuccNet tracklet-supervised (TrackletNet D + A)
WANDB_API_KEY=$WANDB_API_KEY ~/anaconda3/envs/geometric/bin/python train_regressor_pos_query.py \
  --config cfg-rittersport.yaml \
  --num_gpus $NUM_GPUS \
  --target successor \
  --input_layers rgb+drivable+angles \
  --full-checkpoint /data/autograph/checkpoints/local_run_full/e-003.pth \
  --dataset_name test-austin/tracklets_joint -d
