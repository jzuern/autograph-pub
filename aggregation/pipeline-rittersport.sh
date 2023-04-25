export PYTHONPATH=$PYTHONPATH:/home/zuern/self-supervised-graph

# iterate over all cities
#CITIES=(pittsburgh washington paloalto austin detroit miami)
CITIES=(austin)
#SOURCES=(lanegraph tracklets_raw tracklets_joint)

for CITY in "${CITIES[@]}"; do
      ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --city_name $CITY\
                                                             --out_path_root /data/autograph/austin-2404-rittersport/lanegraph/$CITY\
                                                             --sat_image_root /data/lanegraph/woven-data\
                                                             --source lanegraph\
                                                             --max_num_samples 100000 &

                                                                   ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --city_name $CITY\
                                                             --out_path_root /data/autograph/austin-2404-rittersport/tracklets_raw/$CITY\
                                                             --sat_image_root /data/lanegraph/woven-data\
                                                             --source tracklets_raw\
                                                             --max_num_samples 100000 &

                                                                   ~/anaconda3/envs/geometric/bin/python aggregate_av2.py --city_name $CITY\
                                                             --out_path_root /data/autograph/austin-2404-rittersport/tracklets_joint/$CITY\
                                                             --sat_image_root /data/lanegraph/woven-data\
                                                             --source tracklets_joint\
                                                             --max_num_samples 100000 &
    done
    wait
done



#ps aux | grep aggregate_av2.py | awk '{print $2}' | xargs kill -9