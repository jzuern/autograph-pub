export PYTHONPATH=$PYTHONPATH:/home/zuern/self-supervised-graph

# image sizes
# austin      3516 MP
# detroit     1900 MP
# miami       5876 MP
# paloalto    2226 MP
# pittsburgh  2756 MP
# washington  5136 MP

# iterate over all cities
#CITIES=(pittsburgh austin detroit paloalto washington miami)
CITIES=(austin paloalto miami)
#SOURCES=(lanegraph tracklets_joint tracklets_raw)
SOURCES=(tracklets_joint tracklets_raw)

NUM_PARALLEL=4
NUM_PARSING=2

run_train () {
  for CITY in "${CITIES[@]}"; do
    for SOURCE in "${SOURCES[@]}"; do
      for ((tid=1; tid<=NUM_PARALLEL; tid++)); do
        echo "Processing $CITY, $tid / $NUM_PARALLEL !"
        ~/anaconda3/envs/geometric/bin/python aggregate_av2.py \
          --city_name $CITY\
          --out_path_root /data/autograph/all-3004/$SOURCE/$CITY\
          --urbanlanegraph_root /data/lanegraph/urbanlanegraph-dataset-dev\
          --source $SOURCE \
          --max_num_samples 5000 \
          --num_parallel $NUM_PARALLEL \
          --thread_id $tid &
        sleep 600 # sleep to give time start generating
      done
      wait
    done
    wait
  done
  wait
}



for ((i=1; i<=NUM_PARSING; i++)); do
  echo "Parsing $i / $NUM_PARSING !"
  run_train &
  sleep 600 # sleep to give time start generating
done




#ps aux | grep aggregate_av2.py | awk '{print $2}' | xargs kill -9