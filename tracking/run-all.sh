
#export PYTHON=/home/zuern/anaconda3/envs/centerpoint/bin/python
export PYTHON=/home/zuern/anaconda3/envs/lanegcn/bin/python

# DETECT
export CKPT=/home/zuern/Downloads/cbgs_voxel0075_centerpoint_nds_6648.pth
export CFG=/home/zuern/self-supervised-graph/tracking/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml


(cd OpenPCDet/tools && $PYTHON demo.py --cfg_file $CFG --ckpt $CKPT --ext .feather --av2_root /home/zuern/datasets/argoverse2-full/lidar/test/)  # (2000 samples)



#(cd OpenPCDet/tools && $PYTHON demo.py --cfg_file $CFG --ckpt $CKPT --ext .feather --av2_root /home/zuern/datasets/argoverse2-full/sensor/test/)  # (150 samples) done
#(cd OpenPCDet/tools && $PYTHON demo.py --cfg_file $CFG --ckpt $CKPT --ext .feather --av2_root /home/zuern/datasets/argoverse2-full/sensor/val/)  # (700 samples) done
#(cd OpenPCDet/tools && $PYTHON demo.py --cfg_file $CFG --ckpt $CKPT --ext .feather --av2_root /home/zuern/datasets/argoverse2-full/sensor/train/)  # (150 samples) done
#
#(cd OpenPCDet/tools && $PYTHON demo.py --cfg_file $CFG --ckpt $CKPT --ext .feather --av2_root /home/zuern/datasets/argoverse2-full/lidar/test/)  # (2000 samples)
#(cd OpenPCDet/tools && $PYTHON demo.py --cfg_file $CFG --ckpt $CKPT --ext .feather --av2_root /home/zuern/datasets/argoverse2-full/lidar/val/)    # (2000 samples) done
#(cd OpenPCDet/tools && $PYTHON demo.py --cfg_file $CFG --ckpt $CKPT --ext .feather --av2_root /home/zuern/datasets/argoverse2-full/lidar/train/) # (20000 samples)
#
#(cd OpenPCDet/tools && $PYTHON demo.py --cfg_file $CFG --ckpt $CKPT --ext .feather --av2_root /home/zuern/datasets/argoverse2-full/tbv/) # (1043 samples)
#
#
## TRACK
#$PYTHON tracking.py --av2_root /home/zuern/datasets/argoverse2-full/sensor/test/ # done
#$PYTHON tracking.py --av2_root /home/zuern/datasets/argoverse2-full/sensor/val/ # done
#$PYTHON tracking.py --av2_root /home/zuern/datasets/argoverse2-full/sensor/train/ # done
#
#$PYTHON tracking.py --av2_root /home/zuern/datasets/argoverse2-full/lidar/test/
#$PYTHON tracking.py --av2_root /home/zuern/datasets/argoverse2-full/lidar/val/
#$PYTHON tracking.py --av2_root /home/zuern/datasets/argoverse2-full/lidar/train/
#
#$PYTHON tracking.py --av2_root /home/zuern/datasets/argoverse2-full/tbv/
#
#
#find /home/zuern/datasets/argoverse2-full -type f -wholename '/home/zuern/datasets/argoverse2-full/sensor/*/*/tracking.pickle' >> /home/zuern/tracking/tracking_files.txt
#find /home/zuern/datasets/argoverse2-full -type f -wholename '/home/zuern/datasets/argoverse2-full/tbv/*/tracking.pickle' >> /home/zuern/tracking/tracking_files.txt
