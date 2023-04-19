export PYTHONPATH=$PYTHONPATH:/home/zuern/self-supervised-graph

# iterate over all cities
#CITIES=(pittsburgh washington paloalto austin detroit miami)
CITIES=(austin)
SOURCES=(lanegraph tracklets_raw tracklets_joint)
NUM_PARALLEL=1

for CITY in "${CITIES[@]}"; do
  for SOURCE in "${SOURCES[@]}"; do
    for ((tid=1; tid<=NUM_PARALLEL; tid++)); do
      echo "Processing $CITY, $tid / $NUM_PARALLEL !"
      ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --city_name $CITY\
                                                             --out_path_root /data/autograph/1804/$SOURCE/$CITY\
                                                             --sat_image_root /data/lanegraph/woven-data\
                                                             --source $SOURCE\
                                                             --max_num_samples 10000\
                                                             --num_parallel $NUM_PARALLEL\
                                                             --thread_id $tid &
      sleep 120 # sleep to give time start generating
    done
    wait
  done
  wait
done
wait

#ps aux | grep aggregate_av2.py | awk '{print $2}' | xargs kill -9