export PYTHONPATH=$PYTHONPATH:/home/zuern/self-supervised-graph

# iterate over all cities
#CITIES=(pittsburgh washington paloalto austin detroit miami)
CITIES=(austin)
NUM_PARALLEL=1

for CITY in "${CITIES[@]}"; do
  for ((tid=1; tid<=NUM_PARALLEL; tid++)); do
    echo "Processing $CITY, $tid / $NUM_PARALLEL !"
    ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --city_name $CITY\
                                                           --out_path_root /data/autograph/test-all-cities/$CITY/lanegraph \
                                                           --sat_image_root /data/lanegraph/woven-data \
                                                           --source lanegraph \
                                                           --max_num_samples 1000 \
                                                           --num_parallel $NUM_PARALLEL \
                                                           --thread_id $tid &

                                         ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --city_name $CITY\
                                                           --out_path_root /data/autograph/test-all-cities/$CITY/tracklets_raw \
                                                           --sat_image_root /data/lanegraph/woven-data \
                                                           --source tracklets_raw \
                                                           --max_num_samples 1000 \
                                                           --num_parallel $NUM_PARALLEL \
                                                           --thread_id $tid &

                                       ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --city_name $CITY\
                                                           --out_path_root /data/autograph/test-all-cities/$CITY/tracklets_joint \
                                                           --sat_image_root /data/lanegraph/woven-data \
                                                           --source tracklets_joint \
                                                           --max_num_samples 1000 \
                                                           --num_parallel $NUM_PARALLEL \
                                                           --thread_id $tid &
    sleep 60 # sleep X sec to give time start generating
  done
  wait
done
wait

#ps aux | grep aggregate_av2.py | awk '{print $2}' | xargs kill -9