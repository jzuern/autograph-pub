from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from nuscenes import NuScenes
import json
import time
from nuscenes.utils import splits
import numpy as np
import copy



def greedy_assignment(dist):
  matched_indices = []
  if dist.shape[1] == 0:
    return np.array(matched_indices, np.int32).reshape(-1, 2)
  for i in range(dist.shape[0]):
    j = dist[i].argmin()
    if dist[i][j] < 1e16:
      dist[:, j] = 1e18
      matched_indices.append([i, j])
  return np.array(matched_indices, np.int32).reshape(-1, 2)

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck'
]

# 99.9 percentile of the l2 velocity error distribution (per clss / 0.5 second)
# This is an earlier statistcs and I didn't spend much time tuning it.
# Tune this for your model should provide some considerable AMOTA improvement
NUSCENE_CLS_VELOCITY_ERROR = {
    'car': 4,
    'truck': 4,
    'bus': 5.5,
    'trailer': 3,
    'pedestrian': 1,
    'motorcycle': 13,
    'bicycle': 3,
}


class Tracker(object):
    def __init__(self, hungarian=False, max_age=0):
        self.hungarian = hungarian
        self.max_age = max_age

        print("Use hungarian: {}".format(hungarian))

        self.NUSCENE_CLS_VELOCITY_ERROR = NUSCENE_CLS_VELOCITY_ERROR

        self.reset()

    def reset(self):
        self.id_count = 0
        self.tracks = []

    def step_centertrack(self, results, time_lag):
        if len(results) == 0:
            self.tracks = []
            return []
        else:
            temp = []
            for det in results:
                # filter out classes not evaluated for tracking
                if det['detection_name'] not in NUSCENES_TRACKING_NAMES:
                    continue

                det['ct'] = np.array(det['translation'][:2])
                det['tracking'] = np.array(det['velocity'][:2]) * -1 * time_lag
                det['label_preds'] = NUSCENES_TRACKING_NAMES.index(det['detection_name'])
                temp.append(det)

            results = temp

        N = len(results)
        M = len(self.tracks)

        # N X 2
        if 'tracking' in results[0]:
            dets = np.array(
                [det['ct'] + det['tracking'].astype(np.float32)
                 for det in results], np.float32)
        else:
            dets = np.array(
                [det['ct'] for det in results], np.float32)

        item_cat = np.array([item['label_preds'] for item in results], np.int32)  # N
        track_cat = np.array([track['label_preds'] for track in self.tracks], np.int32)  # M

        max_diff = np.array([self.NUSCENE_CLS_VELOCITY_ERROR[box['detection_name']] for box in results], np.float32)

        tracks = np.array(
            [pre_det['ct'] for pre_det in self.tracks], np.float32)  # M x 2

        if len(tracks) > 0:  # NOT FIRST FRAME
            dist = (((tracks.reshape(1, -1, 2) - \
                      dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M
            dist = np.sqrt(dist)  # absolute distance in meter

            invalid = ((dist > max_diff.reshape(N, 1)) + \
                       (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0

            dist = dist + invalid * 1e18
            if self.hungarian:
                dist[dist > 1e18] = 1e18
                matched_indices = linear_assignment(copy.deepcopy(dist))
            else:
                matched_indices = greedy_assignment(copy.deepcopy(dist))
        else:  # first few frame
            assert M == 0
            matched_indices = np.array([], np.int32).reshape(-1, 2)

        unmatched_dets = [d for d in range(dets.shape[0]) \
                          if not (d in matched_indices[:, 0])]

        unmatched_tracks = [d for d in range(tracks.shape[0]) \
                            if not (d in matched_indices[:, 1])]

        if self.hungarian:
            matches = []
            for m in matched_indices:
                if dist[m[0], m[1]] > 1e16:
                    unmatched_dets.append(m[0])
                else:
                    matches.append(m)
            matches = np.array(matches).reshape(-1, 2)
        else:
            matches = matched_indices

        ret = []
        for m in matches:
            track = results[m[0]]
            track['tracking_id'] = self.tracks[m[1]]['tracking_id']
            track['age'] = 1
            track['active'] = self.tracks[m[1]]['active'] + 1
            ret.append(track)

        for i in unmatched_dets:
            track = results[i]
            self.id_count += 1
            track['tracking_id'] = self.id_count
            track['age'] = 1
            track['active'] = 1
            ret.append(track)

        # still store unmatched tracks if its age doesn't exceed max_age, however, we shouldn't output
        # the object in current frame
        for i in unmatched_tracks:
            track = self.tracks[i]
            if track['age'] < self.max_age:
                track['age'] += 1
                track['active'] = 0
                ct = track['ct']

                # movement in the last second
                if 'tracking' in track:
                    offset = track['tracking'] * -1  # move forward
                    track['ct'] = ct + offset
                ret.append(track)

        self.tracks = ret
        return ret



def parse_args():
    parser = argparse.ArgumentParser(description="Tracking Evaluation")
    parser.add_argument("--work_dir", help="the dir to save logs and tracking results")
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from"
    )
    parser.add_argument("--hungarian", action='store_true')
    parser.add_argument("--root", type=str, default="/data/nuscenes")
    parser.add_argument("--version", type=str, default='v1.0-mini')
    parser.add_argument("--max_age", type=int, default=3)

    args = parser.parse_args()

    return args


def save_first_frame():
    args = parse_args()
    nusc = NuScenes(version=args.version,
                    dataroot=args.root,
                    verbose=True)

    if args.version == 'v1.0-trainval':
        scenes = splits.val
    elif args.version == 'v1.0-test':
        scenes = splits.test
    elif args.version == 'v1.0-mini':
        scenes = splits.test
    else:
        raise ValueError("unknown")

    frames = []
    for sample in nusc.sample:
        scene_name = nusc.get("scene", sample['scene_token'])['name']
        if scene_name not in scenes:
            continue

        timestamp = sample["timestamp"] * 1e-6
        token = sample["token"]
        frame = {}
        frame['token'] = token
        frame['timestamp'] = timestamp

        # start of a sequence
        if sample['prev'] == '':
            frame['first'] = True
        else:
            frame['first'] = False
        frames.append(frame)

    del nusc

    res_dir = os.path.join(args.work_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    with open(os.path.join(args.work_dir, 'frames_meta.json'), "w") as f:
        json.dump({'frames': frames}, f)



if __name__ == '__main__':

    args = parse_args()
    print('Deploy OK')

    tracker = Tracker(max_age=args.max_age,
                      hungarian=True)

    with open(args.checkpoint, 'rb') as f:
        predictions = json.load(f)['results']

    with open(os.path.join(args.work_dir, 'frames_meta.json'), 'rb') as f:
        frames = json.load(f)['frames']

    nusc_annos = {
        "results": {},
        "meta": None,
    }

    size = len(frames)

    print("Begin Tracking\n")
    start = time.time()
    for i in range(size):
        token = frames[i]['token']

        # reset tracking after one video sequence
        if frames[i]['first']:
            # use this for sanity check to ensure your token order is correct
            # print("reset ", i)
            tracker.reset()
            last_time_stamp = frames[i]['timestamp']

        time_lag = (frames[i]['timestamp'] - last_time_stamp)
        last_time_stamp = frames[i]['timestamp']

        preds = predictions[token]

        outputs = tracker.step_centertrack(preds, time_lag)
        annos = []

        for item in outputs:
            if item['active'] == 0:
                continue
            nusc_anno = {
                "sample_token": token,
                "translation": item['translation'],
                "size": item['size'],
                "rotation": item['rotation'],
                "velocity": item['velocity'],
                "tracking_id": str(item['tracking_id']),
                "tracking_name": item['detection_name'],
                "tracking_score": item['detection_score'],
            }
            annos.append(nusc_anno)
        nusc_annos["results"].update({token: annos})

    end = time.time()

    second = (end - start)

    speed = size / second
    print("The speed is {} FPS".format(speed))

    nusc_annos["meta"] = {
        "use_camera": False,
        "use_lidar": True,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }

    res_dir = os.path.join(args.work_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    with open(os.path.join(args.work_dir, 'tracking_result.json'), "w") as f:
        json.dump(nusc_annos, f)

