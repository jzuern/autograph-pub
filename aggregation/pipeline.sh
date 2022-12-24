# Process pittsburgh
python aggregate_av2.py --city_name Pittsburgh --out_path_root /data/autograph/preprocessed/pittsburgh
python infer_regressor.py --out_path_root /data/autograph/preprocessed/pittsburgh
python aggregate_av2.py --city_name Pittsburgh --out_path_root /data/autograph/preprocessed/pittsburgh-filtered --export_final