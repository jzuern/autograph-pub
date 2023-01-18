
export PYTHON=/home/zuern/anaconda3/envs/centerpoint/bin/python

# DETECT
export CKPT=/home/zuern/Downloads/cbgs_voxel0075_centerpoint_nds_6648.pth
export CFG=/home/zuern/self-supervised-graph/tracking/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml

(cd OpenPCDet/tools && $PYTHON demo.py --cfg_file $CFG --ckpt $CKPT --ext .feather --av2_root /data/argoverse2-full/sensor/test/)
(cd OpenPCDet/tools && $PYTHON demo.py --cfg_file $CFG --ckpt $CKPT --ext .feather --av2_root /data/argoverse2-full/sensor/val/)
(cd OpenPCDet/tools && $PYTHON demo.py --cfg_file $CFG --ckpt $CKPT --ext .feather --av2_root /data/argoverse2-full/sensor/train/)


# TRACK
$PYTHON tracking.py --av2_root /data/argoverse2-full/sensor/test/
$PYTHON tracking.py --av2_root /data/argoverse2-full/sensor/val/
$PYTHON tracking.py --av2_root /data/argoverse2-full/sensor/train/