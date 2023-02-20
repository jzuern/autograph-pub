export PYTHONPATH=$PYTHONPATH:/home/zuern/self-supervised-graph

# iterate over all cities
CITIES=(miami washington austin detroit pittsburgh paloalto)
#CITIES=(pittsburgh)

EXP="1302"
OUT_DIR="/data/autograph"


NUM_PARALLEL=4


for CITY in "${CITIES[@]}"; do
    echo "Processing $CITY !"
    for ((i=0; i<NUM_PARALLEL; i++)); do
        echo "Processing $CITY, $i !"
        ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --city_name $CITY --out_path_root $OUT_DIR/$EXP/$CITY-pre --source tracklets_dense --num_cpus 1 &
    done
    wait

done
