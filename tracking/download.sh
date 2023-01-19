
# download sensor dataset
#s5cmd --no-sign-request -numworkers 128 cp --exclude "*/cameras/*" "s3://argoai-argoverse/av2/sensor/test/*" /data/argoverse2-full/sensor/test
#s5cmd --no-sign-request -numworkers 128 cp --exclude "*/cameras/*" "s3://argoai-argoverse/av2/sensor/val/*" /data/argoverse2-full/sensor/val
#s5cmd --no-sign-request -numworkers 128 cp --exclude "*/cameras/*" "s3://argoai-argoverse/av2/sensor/train/*" /data/argoverse2-full/sensor/train


# download motion-forecasting dataset
s5cmd --no-sign-request -numworkers 128 cp --exclude "*/cameras/*" "s3://argoai-argoverse/av2/motion-forecasting/test/*" /data/argoverse2-full/motion-forecasting/test
s5cmd --no-sign-request -numworkers 128 cp --exclude "*/cameras/*" "s3://argoai-argoverse/av2/motion-forecasting/val/*" /data/argoverse2-full/motion-forecasting/val
s5cmd --no-sign-request -numworkers 128 cp --exclude "*/cameras/*" "s3://argoai-argoverse/av2/motion-forecasting/train/*" /data/argoverse2-full/motion-forecasting/train

# download tbv dataset
s5cmd --no-sign-request -numworkers 128 cp --exclude "*/cameras/*" "s3://argoai-argoverse/av2/tbv/test/*" /data/argoverse2-full/sensor/test
s5cmd --no-sign-request -numworkers 128 cp --exclude "*/cameras/*" "s3://argoai-argoverse/av2/tbv/val/*" /data/argoverse2-full/sensor/val
s5cmd --no-sign-request -numworkers 128 cp --exclude "*/cameras/*" "s3://argoai-argoverse/av2/tbv/train/*" /data/argoverse2-full/sensor/train



# download lidar dataset
#s5cmd --no-sign-request -numworkers 128 cp --exclude "*/cameras/*" "s3://argoai-argoverse/av2/lidar/test/*" /data/argoverse2-full/lidar/test
#s5cmd --no-sign-request -numworkers 128 cp --exclude "*/cameras/*" "s3://argoai-argoverse/av2/lidar/val/*" /data/argoverse2-full/lidar/val
s5cmd --no-sign-request -numworkers 128 cp --exclude "*/cameras/*" "s3://argoai-argoverse/av2/lidar/train/*" /data/argoverse2-full/lidar/train