export PYTHONPATH=$PYTHONPATH:/home/zuern/self-supervised-graph

# iterate over all cities
CITIES=(pittsburgh miami washington paloalto austin detroit)
#CITIES=(pittsburgh)


for CITY in "${CITIES[@]}"; do
  echo "Processing $CITY !"
  ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --city_name $CITY --out_path_root /home/zuern/datasets/autograph/1502/$CITY --sat_image_root /home/zuern/datasets/lanegraph/ --source tracklets_dense --num_cpus 1 &
  ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --city_name $CITY --out_path_root /home/zuern/datasets/autograph/1502/$CITY --sat_image_root /home/zuern/datasets/lanegraph/ --source tracklets_dense --num_cpus 1 &
  ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --city_name $CITY --out_path_root /home/zuern/datasets/autograph/1502/$CITY --sat_image_root /home/zuern/datasets/lanegraph/ --source tracklets_dense --num_cpus 1 &
  ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --city_name $CITY --out_path_root /home/zuern/datasets/autograph/1502/$CITY --sat_image_root /home/zuern/datasets/lanegraph/ --source tracklets_dense --num_cpus 1 &
  ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --city_name $CITY --out_path_root /home/zuern/datasets/autograph/1502/$CITY --sat_image_root /home/zuern/datasets/lanegraph/ --source tracklets_dense --num_cpus 1 &
  ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --city_name $CITY --out_path_root /home/zuern/datasets/autograph/1502/$CITY --sat_image_root /home/zuern/datasets/lanegraph/ --source tracklets_dense --num_cpus 1 &
  ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --city_name $CITY --out_path_root /home/zuern/datasets/autograph/1502/$CITY --sat_image_root /home/zuern/datasets/lanegraph/ --source tracklets_dense --num_cpus 1 &
  ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --city_name $CITY --out_path_root /home/zuern/datasets/autograph/1502/$CITY --sat_image_root /home/zuern/datasets/lanegraph/ --source tracklets_dense --num_cpus 1
done

EXP="exp-10-01-23"
OUT_DIR="/home/zuern/datasets/autograph"



# kill all provesses with name "aggregate_av2.py"
#ps aux | grep aggregate_av2.py | awk '{print $2}' | xargs kill -9


#for CITY in "${CITIES[@]}"; do
#    echo "Processing $CITY !"
#    ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --sat_image_root /home/zuern/datasets/lanegraph/ --city_name $CITY --out_path_root $OUT_DIR/$EXP/$CITY-pre --source tracklets_sparse
#    ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --sat_image_root /home/zuern/datasets/lanegraph/ --city_name $CITY --out_path_root $OUT_DIR/$EXP/$CITY-pre --source tracklets_dense
#    ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --sat_image_root /home/zuern/datasets/lanegraph/ --city_name $CITY --out_path_root $OUT_DIR/$EXP/$CITY-pre --source lanes
#
#    ~/anaconda3/envs/geometric/bin/python infer_regressor.py --out_path_root $OUT_DIR/$EXP/$CITY-pre --checkpoint ../checkpoints/pit/regressor_cosmic-sponge-2_0040-0.41017.pth
#
#    ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --sat_image_root /home/zuern/datasets/lanegraph/ --city_name $CITY --out_path_root $OUT_DIR/$EXP/$CITY-post --export_final --source tracklets_sparse
#    ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --sat_image_root /home/zuern/datasets/lanegraph/ --city_name $CITY --out_path_root $OUT_DIR/$EXP/$CITY-post --export_final --source tracklets_dense
#    ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --sat_image_root /home/zuern/datasets/lanegraph/ --city_name $CITY --out_path_root $OUT_DIR/$EXP/$CITY-post --export_final --source lanes
#done



