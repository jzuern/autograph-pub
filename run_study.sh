# baseline
CUDA_VISIBLE_DEVICES=0 ~/lanegraph/mpenv36/bin/python train_lanegnn.py --config cfg-rlgpu.yaml --target sparse --in_layers rgb,sdf --gnn_depth 1 &

# vary target
CUDA_VISIBLE_DEVICES=1 ~/lanegraph/mpenv36/bin/python train_lanegnn.py --config cfg-rlgpu.yaml --target dense --in_layers rgb,sdf --gnn_depth 1 &
CUDA_VISIBLE_DEVICES=3 ~/lanegraph/mpenv36/bin/python train_lanegnn.py --config cfg-rlgpu.yaml --target lanes --in_layers rgb,sdf --gnn_depth 1

# vary gnn depth
CUDA_VISIBLE_DEVICES=0 ~/lanegraph/mpenv36/bin/python train_lanegnn.py --config cfg-rlgpu.yaml --target sparse --in_layers rgb,sdf --gnn_depth 2 &
CUDA_VISIBLE_DEVICES=1 ~/lanegraph/mpenv36/bin/python train_lanegnn.py --config cfg-rlgpu.yaml --target sparse --in_layers rgb,sdf --gnn_depth 3 &
CUDA_VISIBLE_DEVICES=3 ~/lanegraph/mpenv36/bin/python train_lanegnn.py --config cfg-rlgpu.yaml --target sparse --in_layers rgb,sdf --gnn_depth 0

# vary input layers
CUDA_VISIBLE_DEVICES=0 ~/lanegraph/mpenv36/bin/python train_lanegnn.py --config cfg-rlgpu.yaml --target sparse --in_layers rgb,sdf,angles --gnn_depth 1 &
CUDA_VISIBLE_DEVICES=0 ~/lanegraph/mpenv36/bin/python train_lanegnn.py --config cfg-rlgpu.yaml --target sparse --in_layers rgb --gnn_depth 1 &
