export PYTHONPATH=$PYTHONPATH:/home/zuern/self-supervised-graph

# iterate over all cities
#city_names=(pittsburgh miami washington paloalto austin detroit)
city_names=(pittsburgh)


for city_name in "${city_names[@]}"; do
    echo "!!!!!!!!!!Processing $city_name!!!!!!!!!!!!!!!"
    ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --city_name $city_name --out_path_root /data/autograph/sparse/$city_name-pre
    ~/anaconda3/envs/geometric/bin/python infer_regressor.py --out_path_root /data/autograph/sparse/$city_name-pre --checkpoint ../checkpoints/pit/regressor_local_run_0520-0.42925.pth
    ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --city_name $city_name --out_path_root /data/autograph/sparse/$city_name-post --export_final
done


# train the model
#~/anaconda3/envs/geometric/bin/python train_lanegnn.py --config cfg-rittersport.yaml
