export AV2ROOT=/home/zuern/datasets/argoverse2-full
export NUMW=256

# download sensor dataset
s5cmd --no-sign-request -numworkers $NUMW cp --exclude "*/cameras/*" "s3://argoai-argoverse/av2/sensor/test/*" $AV2ROOT/sensor/test
s5cmd --no-sign-request -numworkers $NUMW cp --exclude "*/cameras/*" "s3://argoai-argoverse/av2/sensor/val/*" $AV2ROOT/sensor/val
s5cmd --no-sign-request -numworkers $NUMW cp --exclude "*/cameras/*" "s3://argoai-argoverse/av2/sensor/train/*" $AV2ROOT/sensor/train

# download motion-forecasting dataset
s5cmd --no-sign-request -numworkers $NUMW cp --exclude "*/cameras/*" "s3://argoai-argoverse/av2/motion-forecasting/test/*" $AV2ROOT/motion-forecasting/test
s5cmd --no-sign-request -numworkers $NUMW cp --exclude "*/cameras/*" "s3://argoai-argoverse/av2/motion-forecasting/val/*" $AV2ROOT/motion-forecasting/val
s5cmd --no-sign-request -numworkers $NUMW cp --exclude "*/cameras/*" "s3://argoai-argoverse/av2/motion-forecasting/train/*" $AV2ROOT/motion-forecasting/train

# download tbv dataset
s5cmd --no-sign-request -numworkers $NUMW cp --exclude "*/cameras/*" "s3://argoai-argoverse/av2/tbv/*" $AV2ROOT/tbv

# download lidar dataset
s5cmd --no-sign-request -numworkers $NUMW cp --exclude "*/cameras/*" "s3://argoai-argoverse/av2/lidar/test/*" $AV2ROOT/lidar/test
s5cmd --no-sign-request -numworkers $NUMW cp --exclude "*/cameras/*" "s3://argoai-argoverse/av2/lidar/val/*" $AV2ROOT/lidar/val
s5cmd --no-sign-request -numworkers $NUMW cp --exclude "*/cameras/*" "s3://argoai-argoverse/av2/lidar/train/*" $AV2ROOT/lidar/train