import os
from pathlib import Path
import random
from tqdm import tqdm


paths = [
    '/data/autograph/paloalto-2604/',
    '/data/autograph/all-small-2804/',
    #'/data/autograph/all-3004/',
]

splits = [
    'train',
    'eval',
    'test',
    ]

for path in paths:
    for split in splits:

        print("Looking for files in {}".format(path))

        # get all files
        p = Path(path)

        filelist = [str(f) for f in p.glob('**/*') if f.is_file()]
        filelist = sorted(filelist)

        rgb_files = [f for f in filelist if "-rgb.png" in f and split in f]
        sdf_files = [f for f in filelist if "-masks.png" in f and split in f]
        angles_files = [f for f in filelist if "-angles.png" in f and split in f]
        pos_enc_files = [f for f in filelist if "-pos-encoding.png" in f and split in f]
        drivable_gt_files = [f for f in filelist if "-drivable-gt.png" in f and split in f]

        print(len(sdf_files), len(angles_files), len(rgb_files), len(pos_enc_files), len(drivable_gt_files))

        if len(sdf_files) == 0:
            raise ValueError("No files found in {}".format(path))


        # # jointly shuffle them
        c = list(zip(sdf_files, angles_files, rgb_files, pos_enc_files, drivable_gt_files))
        random.shuffle(c)
        sdf_files, angles_files, rgb_files, pos_enc_files, drivable_gt_files = zip(*c)

        # check if all files are present
        assert len(sdf_files) == len(rgb_files)

        # Now we can share the files between type branch and straight
        rgb_branch = [i for i in rgb_files if "branching" in i]
        sdf_branch = [i for i in sdf_files if "branching" in i]
        pos_enc_branch = [i for i in pos_enc_files if "branching" in i]
        drivable_gt_branch = [i for i in drivable_gt_files if "branching" in i]
        angles_branch = [i for i in angles_files if "branching" in i]

        rgb_straight = [i for i in rgb_files if "straight" in i]
        sdf_straight = [i for i in sdf_files if "straight" in i]
        pos_enc_straight = [i for i in pos_enc_files if "straight" in i]
        drivable_gt_straight = [i for i in drivable_gt_files if "straight" in i]
        angles_straight = [i for i in angles_files if "straight" in i]

        print("     Total Branch: {} files".format(len(rgb_branch)))
        print("     Total Straight: {} files".format(len(rgb_straight)))



        if len(rgb_straight) > 2 * len(rgb_branch):
            to_delete = len(rgb_straight) - 2 * len(rgb_branch)
            print("     Deleting {} straight files".format(to_delete))

            for i in tqdm(range(to_delete), total=to_delete):
                os.remove(rgb_straight[i])
                os.remove(sdf_straight[i])
                os.remove(pos_enc_straight[i])
                os.remove(drivable_gt_straight[i])
                os.remove(angles_straight[i])

        else:
            print("     Nothing to delete")


print("Done")