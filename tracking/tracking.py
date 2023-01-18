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
import pickle
from scipy.optimize import linear_sum_assignment
from glob import glob
import matplotlib.pyplot as plt


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
    1, # car
]

# 99.9 percentile of the l2 velocity error distribution (per clss / 0.5 second)
# This is an earlier statistcs and I didn't spend much time tuning it.
# Tune this for your model should provide some considerable AMOTA improvement
# NUSCENE_CLS_VELOCITY_ERROR = {
#     'car': 4,
#     'truck': 4,
#     'bus': 5.5,
#     'trailer': 3,
#     'pedestrian': 1,
#     'motorcycle': 13,
#     'bicycle': 3,
# }
NUSCENE_CLS_VELOCITY_ERROR = {
    1: 4, # car
}


SCORE_THRSHLD = 0.3

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
            detections = []

            for i in range(len(results["pred_labels"])):

                detection = {}
                detection["label_preds"] = results["pred_labels"][i]
                detection["score"] = results["pred_scores"][i]

                box = results["pred_boxes"][i]
                # format:
                # center = box[0:3]
                # lwh = box[3:6]
                # axis_angles = np.array([0, 0, box[6] + 1e-10])
                detection["translation"] = box[0:3]
                detection["velocity"] = np.zeros([3])
                detection["size"] = box[3:6]
                detection["rotation"] = box[6]

                detections.append(detection)


            for det in detections:
                # filter out classes not evaluated for tracking
                if det['label_preds'] not in NUSCENES_TRACKING_NAMES:
                    continue

                # filter out low-score predictions
                if det['score'] < SCORE_THRSHLD:
                    continue

                det['ct'] = np.array(det['translation'][:2])
                det['tracking'] = np.array(det['velocity'][:2]) * -1 * time_lag
                temp.append(det)

            results = temp

        if len(results) == 0:
            self.tracks = []
            return []


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

        max_diff = np.array([self.NUSCENE_CLS_VELOCITY_ERROR[box['label_preds']] for box in results], np.float32)

        tracks = np.array(
            [pre_det['ct'] for pre_det in self.tracks], np.float32)  # M x 2

        if len(tracks) > 0:  # NOT FIRST FRAME
            dist = (((tracks.reshape(1, -1, 2) - \
                      dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M
            dist = np.sqrt(dist)  # absolute distance in meter

            invalid = ((dist > max_diff.reshape(N, 1)) + (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0

            dist = dist + invalid * 1e18
            if self.hungarian:
                dist[dist > 1e18] = 1e18
                ind = linear_sum_assignment(copy.deepcopy(dist))
                matched_indices = np.vstack([ind[0], ind[1]]).T
            else:
                matched_indices = greedy_assignment(copy.deepcopy(dist))
        else:  # first few frame
            assert M == 0
            matched_indices = np.array([], np.int32).reshape(-1, 2)

        unmatched_dets = [d for d in range(dets.shape[0]) if not (d in matched_indices[:, 0])]
        unmatched_tracks = [d for d in range(tracks.shape[0]) if not (d in matched_indices[:, 1])]

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
    parser.add_argument("--version", type=str, default='v1.0-mini')
    parser.add_argument("--max_age", type=int, default=3)
    parser.add_argument("--av2_root", type=str)

    args = parser.parse_args()

    return args



if __name__ == '__main__':

    args = parse_args()

    tracker = Tracker(max_age=args.max_age,
                      hungarian=True)


    predictions_files = sorted(glob(args.av2_root + "/*/detections.pickle"))


    for predictions_file in predictions_files:
        print("Performing Tracking on predictions file {}".format(predictions_file))

        tracking_file = predictions_file.replace("detections.pickle", "tracking.pickle")

        if os.path.isfile(tracking_file):
            print("    skip. Existing.")
            continue

        with open(predictions_file, "rb") as f:
            predictions_dict = pickle.load(f)

        av2_annos = {
            "results": {},
            "city_name": predictions_dict["city_name"],
            "meta": None,
        }

        predictions = predictions_dict["predictions"]

        size = len(predictions)


        tracker.reset()
        ego_pos = []

        for i in range(size):
            token = predictions[i]['fname_lidar']
            ego_pos.append(predictions[i]['ego_pos'])

            # reset tracking after one video sequence
            if i == 0:
                # use this for sanity check to ensure your token order is correct
                # print("reset ", i)
                tracker.reset()
                last_time_stamp = predictions[i]['timestamp']

            time_lag = (predictions[i]['timestamp'] - last_time_stamp)
            last_time_stamp = predictions[i]['timestamp']

            outputs = tracker.step_centertrack(predictions[i], time_lag)
            annos = []

            for item in outputs:
                if item['active'] == 0:
                    continue
                anno = {
                    #"sample_token": token,
                    "translation": item['translation'],
                    "size": item['size'],
                    "rotation": item['rotation'],
                    "velocity": item['velocity'],
                    "tracking_id": str(item['tracking_id']),
                    #"tracking_name": item['detection_name'],
                    #"tracking_score": item['detection_score'],
                    "label": item['label_preds'],
                }
                annos.append(anno)
            av2_annos["results"].update({token: annos})

        av2_annos["ego_pos"] = ego_pos

        av2_annos["meta"] = {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }

        with open(tracking_file, "wb") as f:
            pickle.dump(av2_annos, f)


