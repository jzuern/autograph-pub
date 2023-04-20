#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/home/buechner/zuern/self-supervised-graph

# iterate over all cities
#CITIES=(pittsburgh washington paloalto austin detroit miami)
CITIES=(austin)
SOURCES=(tracklets_joint tracklets_raw lanegraph)
NUM_PARALLEL=4

parsing () {

  for CITY in "${CITIES[@]}"; do
    for SOURCE in "${SOURCES[@]}"; do
      for ((tid=1; tid<=NUM_PARALLEL; tid++)); do
        echo "Processing $CITY, $tid / $NUM_PARALLEL !"
        /home/buechner/zuern/geometric/bin/python aggregate_av2.py --city_name $CITY\
                                                               --out_path_root /data/buechner/zuern/autograph/austin-real2/$SOURCE/$CITY \
                                                               --urbanlanegraph_root /home/buechner/zuern/urbanlanegraph-dataset-dev/ \
                                                               --source $SOURCE \
                                                               --max_num_samples 10000 \
                                                               --num_parallel $NUM_PARALLEL \
                                                               --thread_id $tid &
        sleep 60 # sleep to give time start generating
      done
      wait
    done
    wait
  done
  wait
}


NUM_PARSING=10

for ((i=1; i<=NUM_PARSING; i++)); do
  echo "Parsing $i / $NUM_PARSING !"
  parsing &
  sleep 10 # sleep to give time start generating
done



# kill all provesses with name "aggregate_av2.py"
#ps aux | grep aggregate_av2.py | awk '{print $2}' | xargs kill -9


