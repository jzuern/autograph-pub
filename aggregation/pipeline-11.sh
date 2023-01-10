export PYTHONPATH=$PYTHONPATH:/home/zuern/self-supervised-graph

# iterate over all cities
#CITIES=(pittsburgh miami washington paloalto austin detroit)
CITIES=(pittsburgh)

EXP="exp-10-01-23"
OUT_DIR="/home/zuern/datasets/autograph"

for CITY in "${CITIES[@]}"; do
    echo "Processing $CITY !"
    ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --sat_image_root /home/zuern/datasets/lanegraph/ --city_name $CITY --out_path_root $OUT_DIR/$EXP/$CITY-pre --source tracklets_sparse
    ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --sat_image_root /home/zuern/datasets/lanegraph/ --city_name $CITY --out_path_root $OUT_DIR/$EXP/$CITY-pre --source tracklets_dense
    ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --sat_image_root /home/zuern/datasets/lanegraph/ --city_name $CITY --out_path_root $OUT_DIR/$EXP/$CITY-pre --source lanes

    ~/anaconda3/envs/geometric/bin/python infer_regressor.py --out_path_root $OUT_DIR/$EXP/$CITY-pre --checkpoint ../checkpoints/pit/regressor_cosmic-sponge-2_0040-0.41017.pth

    ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --sat_image_root /home/zuern/datasets/lanegraph/ --city_name $CITY --out_path_root $OUT_DIR/$EXP/$CITY-post --export_final --source tracklets_sparse
    ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --sat_image_root /home/zuern/datasets/lanegraph/ --city_name $CITY --out_path_root $OUT_DIR/$EXP/$CITY-post --export_final --source tracklets_dense
    ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --sat_image_root /home/zuern/datasets/lanegraph/ --city_name $CITY --out_path_root $OUT_DIR/$EXP/$CITY-post --export_final --source lanes
done
