import os

# Get stats of dataset (e.g. number of images, number of classes, etc.)


root_dir = '/data/autograph/all-3004/'


# get list of files in all subdirectories
filelist = []
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('-rgb.png'):
            filelist.append(os.path.join(root, file))

# get train / eval / test splits:

train_list = [f for f in filelist if '/train/' in f]
eval_list = [f for f in filelist if '/eval/' in f]
test_list = [f for f in filelist if '/test/' in f]


# get tracklets_raw / tracklets_joint and lanegraph

tracklets_raw_list = [f for f in filelist if 'tracklets_raw' in f]
tracklets_joint_list = [f for f in filelist if 'tracklets_joint' in f]
lanegraph_list = [f for f in filelist if 'lanegraph' in f]

# get branching / straight

branching_list = [f for f in filelist if 'branching' in f]
straight_list = [f for f in filelist if 'straight' in f]


# get each city:
austin_list = [f for f in filelist if 'austin' in f]
detroit_list = [f for f in filelist if 'detroit' in f]
miami_list = [f for f in filelist if 'miami' in f]
paloalto_list = [f for f in filelist if 'paloalto' in f]
pittsburgh_list = [f for f in filelist if 'pittsburgh' in f]
washington_list = [f for f in filelist if 'washington' in f]


# print stats

print("Stats for directory: {}\n".format(root_dir))
print('Number of train images: {} ({:.2f} %)'.format(len(train_list), len(train_list) / len(filelist) * 100))
print('Number of eval images: {} ({:.2f} %)'.format(len(eval_list), len(eval_list) / len(filelist) * 100))
print('Number of test images: {} ({:.2f} %)\n'.format(len(test_list), len(test_list) / len(filelist) * 100))

print('Number of tracklets_raw: {} ({:.2f} %)'.format(len(tracklets_raw_list), len(tracklets_raw_list) / len(filelist) * 100))
print('Number of tracklets_joint: {} ({:.2f} %)'.format(len(tracklets_joint_list), len(tracklets_joint_list) / len(filelist) * 100))
print('Number of lanegraph: {} ({:.2f} %)\n'.format(len(lanegraph_list), len(lanegraph_list) / len(filelist) * 100))

print('Number of branching: {} ({:.2f} %)'.format(len(branching_list), len(branching_list) / len(filelist) * 100))
print('Number of straight: {} ({:.2f} %)\n'.format(len(straight_list), len(straight_list) / len(filelist) * 100))

print('Number of austin: {} ({:.2f} %)'.format(len(austin_list), len(austin_list) / len(filelist) * 100))
print('Number of detroit: {} ({:.2f} %)'.format(len(detroit_list), len(detroit_list) / len(filelist) * 100))
print('Number of miami: {} ({:.2f} %)'.format(len(miami_list), len(miami_list) / len(filelist) * 100))
print('Number of paloalto: {} ({:.2f} %)'.format(len(paloalto_list), len(paloalto_list) / len(filelist) * 100))
print('Number of pittsburgh: {} ({:.2f} %)'.format(len(pittsburgh_list), len(pittsburgh_list) / len(filelist) * 100))
print('Number of washington: {} ({:.2f} %)'.format(len(washington_list), len(washington_list) / len(filelist) * 100))




