import os.path
from typing import List
from glob import glob

import numpy as np
import plotly.graph_objs as go
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import plot_maps
import pickle
import matplotlib.pyplot as plt

import tensorflow as tf



def plot_maps_only(maps) -> None:

    feature_list = []
    for map in maps:
        for feature in map:
            feature_list.append(feature)

    # also plot road polylines
    for feature in feature_list:
        if feature.HasField('road_line') or feature.HasField('lane'):

            road_geo = feature.road_line.polyline
            xs = [segment.x for segment in road_geo]
            ys = [segment.y for segment in road_geo]
            zs = [segment.z for segment in road_geo]

            plt.plot(xs, ys, 'r-', linewidth=1)

            lane_geo = feature.lane.polyline
            xs = [segment.x for segment in lane_geo]
            ys = [segment.y for segment in lane_geo]
            zs = [segment.z for segment in lane_geo]

            plt.plot(xs, ys, 'b-', linewidth=1)
    plt.show()



def plot_point_clouds_with_maps(frames: List[dataset_pb2.Frame]) -> None:
    """Plot the point clouds within the given frames with map data.

    Map data must be populated in the first frame in the list.

    Args:
      frames: A list of frames to be plotted, frames[0] must contain map data.
    """

    # Plot the map features.
    if len(frames) == 0:
        return
    figure = plot_maps.plot_map_features(frames[0].map_features)

    for frame in frames:
        # Parse the frame lidar data into range images.
        range_images, camera_projections, seg_labels, range_image_top_poses = (
            frame_utils.parse_range_image_and_camera_projection(frame)
        )

        # Project the range images into points.
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_poses,
            keep_polar_features=True,
        )
        xyz = points[0][:, 3:]
        num_points = xyz.shape[0]

        # Transform the points from the vehicle frame to the world frame.
        xyz = np.concatenate([xyz, np.ones([num_points, 1])], axis=-1)
        transform = np.reshape(np.array(frame.pose.transform), [4, 4])
        xyz = np.transpose(np.matmul(transform, np.transpose(xyz)))[:, 0:3]

        # Correct the pose of the points into the coordinate system of the first
        # frame to align with the map data.
        offset = frame.map_pose_offset
        points_offset = np.array([offset.x, offset.y, offset.z])
        xyz += points_offset

        # Plot the point cloud for this frame aligned with the map data.
        intensity = points[0][:, 0]
        figure.add_trace(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode='markers',
                marker=dict(
                    size=1,
                    color=intensity,  # set color to an array/list of desired values
                    colorscale='Pinkyl',  # choose a colorscale
                    opacity=0.8,
                ),
            )
        )

    figure.show()


# download files: gcloud storage cp "gs://waymo_open_dataset_v_1_4_2/individual_files/training/*.tfrecord" /data/waymo_open_dataset_v_1_4_2/validation


FILENAMES = glob('/data/waymo_open_dataset_v_1_4_2/*/*.tfrecord')

counter = 0

if not os.path.exists("waymo-open-maps.pickle"):
    maps = {}
    maps["filenames"] = []
    # dump
    pickle.dump(maps, open("waymo-open-maps.pickle", "wb"))
else:
    maps = pickle.load(open("waymo-open-maps.pickle", "rb"))


for FILENAME in FILENAMES:
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')

    print(FILENAME)
    if FILENAME in maps["filenames"]:
        print("     already in maps. Skipping and deleting file")
        os.remove(FILENAME)
        continue

    for data in dataset:
        frame = dataset_pb2.Frame.FromString(bytearray(data.numpy()))
        city_name = frame.context.stats.location
        if frame.map_features:
            if city_name not in maps:
                maps[city_name] = [frame.map_features]
            else:
                maps[city_name].append(frame.map_features)
            maps["filenames"].append(FILENAME)
            pickle.dump(maps, open("waymo-open-maps.pickle", "wb"))
            print("     found map features for city: ", city_name, " and saved to maps")

    counter += 1

print(maps.keys())
plot_maps_only(maps["location_sf"])
