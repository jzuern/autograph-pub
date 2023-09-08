# This script copies the files in the global and local directories to a new directory
import os
import shutil


global_dir = "/data/autograph/evaluations/G_agg/tracklets/austin_72_29021_46605/G_agg_global/"
local_dir = "/data/autograph/evaluations/G_agg/tracklets/austin_72_29021_46605/G_agg_local/"
inference_dir = "/data/autograph/evaluations/G_agg/tracklets/austin_72_29021_46605/inference/"


# save all the files in the directory in a list
files = os.listdir(global_dir)
files = sorted(files)

# get the first 4 characters of each file name
# this is the frame number


# now copy the files to a new directory with the frame number as the file name
for f in files:
    print(f)
    #shutil.copy(global_dir + f, '/data/autograph/evaluations/G_agg/tracklets/austin_72_29021_46605/frames_global/' + f)
    #shutil.copy(local_dir + f, '/data/autograph/evaluations/G_agg/tracklets/austin_72_29021_46605/frames_local/' + f)
    shutil.copy(inference_dir + f, '/data/autograph/evaluations/G_agg/tracklets/austin_72_29021_46605/frames_inference/' + f)

