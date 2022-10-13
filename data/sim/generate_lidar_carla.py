import os
import sys
import numpy as np
import queue
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import networkx as nx

sys.path.append('/data/carla/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg')
sys.path.append('/data/carla/PythonAPI/carla/')

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner


import carla2dboundingbox.carla_vehicle_annotator as cva
from carla2dboundingbox.carla_vehicle_annotator import get_camera_intrinsic

from util import rasterize_lidar, clean_dumpster
from spawner import spawn


#########################
# SETTINGS
#########################

use_static_sensors = True
num_vehicles = 200
dump_root_dir = '/data/self-supervised-graph/'
# dump_root_dir = '/SSD/dumps/'

########################################################

def retrieve_data(sensor_queue, frame, timeout=10):
    while True:
        try:
            data = sensor_queue.get(True, timeout)
        except queue.Empty:
            return None
        if data.frame == frame:
            return data

# Create client object
client = carla.Client("0.0.0.0", 2000)
client.set_timeout(10.0)  # seconds

# Create world object
# world = client.get_world()
world = client.load_world('Town03')

map = world.get_map()
waypoint_tuple_list = map.get_topology()

start_waypoints = [x[0] for x in waypoint_tuple_list]
end_waypoints = [x[1] for x in waypoint_tuple_list]

fig, ax = plt.subplots()

G_gt = nx.DiGraph()

for waypoint_tuple in waypoint_tuple_list:
    next_steps = waypoint_tuple[0].next_until_lane_end(5.0)
    random_color = np.random.rand(3,)

    G_gt.add_node(waypoint_tuple[0].transform.location, pos=(waypoint_tuple[0].transform.location.x, waypoint_tuple[0].transform.location.y))
    for next_step in next_steps:
        G_gt.add_node(next_step.transform.location, pos=(next_step.transform.location.x, next_step.transform.location.y))

    # also plot initial arrow
    plt.plot([waypoint_tuple[0].transform.location.x, next_steps[0].transform.location.x],
             [waypoint_tuple[0].transform.location.y, next_steps[0].transform.location.y], color=random_color)
    G_gt.add_edge(waypoint_tuple[0].transform.location, next_steps[0].transform.location)

    for i in range(len(next_steps)-1):
        start = next_steps[i].transform.location
        end = next_steps[i+1].transform.location
        ax.arrow(start.x, start.y, end.x-start.x, end.y-start.y, head_width=0.5, head_length=0.5,
                 fc=random_color, ec=random_color)
        G_gt.add_edge(start, end)

plt.show()

# also show G_gt
locations = nx.get_node_attributes(G_gt, 'pos')
pos = np.array([v for k, v in locations.items()])
nx.draw(G_gt, pos=pos, node_size=10)
plt.show()




# Make fixed time step
settings = world.get_settings()
settings.fixed_delta_seconds = 0.1
settings.synchronous_mode = True
world.apply_settings(settings)

# Weather
weather = world.get_weather()
weather.precipitation = 0.0
weather.precipitation_deposits = 0.0
world.set_weather(weather)

# SETTINGS
min_vehicle_distance = 2.0


# Kill vehicles and walkers
actor_list = world.get_actors()
for vehicle in actor_list.filter('vehicle.*.*'):
    vehicle.destroy()
for walker in actor_list.filter('walker.pedestrian.*'):
    walker.destroy()

# Spawn NPC agents
spawn(client,
      num_vehicles=num_vehicles,
      num_walkers=0)

# Get blueprint library
blueprint_library = world.get_blueprint_library()

# Bookkeeping
q_list = []
idx = 0
tick_queue = queue.Queue()
world.on_tick(tick_queue.put)
q_list.append(tick_queue)
tick_idx = idx
idx = idx + 1


# Spawn ego vehicle driving around
actor_list = world.get_actors().filter('vehicle.*')
my_vehicle = actor_list[0]
my_vehicle.set_autopilot(True)


# Create RGB BEV camera
bev_location = carla.Location(x=0., y=-100, z=200)
static_sensor_location = carla.Location(x=0, y=-100, z=1.5)

cam_transform = carla.Transform(bev_location, carla.Rotation(yaw=0, pitch=-90, roll=-90))

camera_bev_rgb = blueprint_library.find('sensor.camera.rgb')
camera_bev_rgb.set_attribute('image_size_x', '1024')
camera_bev_rgb.set_attribute('image_size_y', '1024')
camera_bev_rgb.set_attribute('fov', '10')
camera_bev_rgb = world.spawn_actor(camera_bev_rgb, cam_transform)
camera_bev_rgb_queue = queue.Queue()
camera_bev_rgb.listen(camera_bev_rgb_queue.put)
q_list.append(camera_bev_rgb_queue)

bev_rgb_idx = idx
idx = idx + 1

# Create semantic BEV camera

camera_bev_sem = blueprint_library.find('sensor.camera.semantic_segmentation')
camera_bev_sem.set_attribute('image_size_x', '1024')
camera_bev_sem.set_attribute('image_size_y', '1024')
camera_bev_sem.set_attribute('fov', '10')
camera_bev_sem = world.spawn_actor(camera_bev_sem, cam_transform)
camera_bev_sem_queue = queue.Queue()
camera_bev_sem.listen(camera_bev_sem_queue.put)
q_list.append(camera_bev_sem_queue)

bev_sem_idx = idx
idx = idx + 1




# Create 4 cameras
cameras = []

# Find the blueprint of the sensor.
camera_blueprint = world.get_blueprint_library().find('sensor.camera.rgb')

camera_distortion = [0, 0, 0, 0, 0, 0]
camera_distortion[0] = camera_blueprint.get_attribute('lens_circle_falloff')
camera_distortion[1] = camera_blueprint.get_attribute('lens_circle_multiplier')
camera_distortion[2] = camera_blueprint.get_attribute('lens_k')
camera_distortion[3] = camera_blueprint.get_attribute('lens_kcube')
camera_distortion[4] = camera_blueprint.get_attribute('lens_x_size')
camera_distortion[5] = camera_blueprint.get_attribute('lens_y_size')


for i in range(0, 1):
    # Modify the attributes of the blueprint to set image resolution and field of view.
    camera_blueprint.set_attribute('image_size_x', '1920')
    camera_blueprint.set_attribute('image_size_y', '1080')
    camera_blueprint.set_attribute('fov', '90')

    if use_static_sensors:
        rot = carla.Rotation(yaw=i*90)
        cam_transform = carla.Transform(static_sensor_location, rot)
        sensor = world.spawn_actor(camera_blueprint, cam_transform,)
    else:
        # Provide the position of the sensor relative to the vehicle.
        cam_transform = carla.Transform(carla.Location(x=0.8, z=1.7, y=-1.0), carla.Rotation(yaw=270))
        sensor = world.spawn_actor(camera_blueprint, cam_transform, attach_to=my_vehicle)


    # Subscribe to the sensor stream by providing a callback function, this function is
    # called each time a new image is generated by the sensor.
    if i == 0:
        # sensor.listen(lambda data: dump0(data))
        cam_queue = queue.Queue()
        sensor.listen(cam_queue.put)
        q_list.append(cam_queue)
        cam_0_idx = idx
        idx = idx + 1

    cameras.append(sensor)
    camera_intrinsics = get_camera_intrinsic(sensor)

    print("Camera {} ready".format(i))


# Create Lidar

# --------------
# Add a new LIDAR sensor to my ego
# --------------
# Spawn LIDAR sensor
lidar_raw_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')

lidar_bp.set_attribute('channels', '64')
lidar_bp.set_attribute('points_per_second', '1120000')
lidar_bp.set_attribute('upper_fov', '10')
lidar_bp.set_attribute('lower_fov', '-30')
lidar_bp.set_attribute('range', '100')
lidar_bp.set_attribute('rotation_frequency', '20')

lidar_raw_bp.set_attribute('channels', '64')
lidar_raw_bp.set_attribute('points_per_second', '1120000')
lidar_raw_bp.set_attribute('upper_fov', '10')
lidar_raw_bp.set_attribute('lower_fov', '-30')
lidar_raw_bp.set_attribute('range', '100')
lidar_raw_bp.set_attribute('rotation_frequency', '20')

if use_static_sensors:
    lidar_transform = carla.Transform(static_sensor_location)
    lidar = world.spawn_actor(lidar_bp, lidar_transform)
    lidar_raw = world.spawn_actor(lidar_raw_bp, lidar_transform)
else:
    lidar = world.spawn_actor(lidar_bp, cam_transform, attach_to=my_vehicle)
    lidar_raw = world.spawn_actor(lidar_raw_bp, cam_transform, attach_to=my_vehicle)

lidar_queue = queue.Queue()
lidar.listen(lidar_queue.put)
q_list.append(lidar_queue)
lidar_idx = idx
idx = idx+1

print('LIDAR ready')

lidar_raw_queue = queue.Queue()
lidar_raw.listen(lidar_raw_queue.put)
q_list.append(lidar_raw_queue)
lidar_raw_idx = idx
idx = idx+1

print('LIDAR raw ready')


actor_list = world.get_actors()

print("----------------------------------------")
print("Starting recording")

# Begin the loop
frame_counter = 0

nowFrame = world.tick()

clean_dumpster(dump_root_dir)


while True:

    print("Frame ", nowFrame)
    frame_counter += 1

    # Make one time step
    nowFrame = world.tick()

    # Skip first few frames to let vehicles settle in
    if frame_counter < 10:
        continue

    # Extract the available data
    data = [retrieve_data(q, nowFrame) for q in q_list]
    assert all(x.frame == nowFrame for x in data if x is not None)

    # Skip if any sensor data is not available
    if None in data:
        print(data)
        continue

    vehicles_raw = world.get_actors().filter('vehicle.*')
    snap = data[tick_idx]

    lidar_img = data[lidar_idx]
    lidar_raw_data = data[lidar_raw_idx]
    img_rgb_bev = data[bev_rgb_idx]
    img_rgb_bev = np.array(img_rgb_bev.raw_data).reshape((img_rgb_bev.height, img_rgb_bev.width, 4))[:, :, :3]

    img_sem_bev = data[bev_sem_idx]
    img_sem_bev = np.array(img_sem_bev.raw_data).reshape((img_sem_bev.height, img_sem_bev.width, 4))[:, :, 2]

    # Attach additional information to the snapshot
    vehicles = cva.snap_processing(vehicles_raw, snap)

    # get relative transofrmation between sensor and vehicles
    _, rel_transform = cva.get_list_transform(vehicles, lidar_raw)
    rel_transform = np.array(rel_transform)
    distances = np.sqrt(rel_transform[:, 0] ** 2 + rel_transform[:, 1] ** 2)

    coords = []

    for vehicle, trans in zip(vehicles, rel_transform):
        x = trans[0]
        y = trans[1]
        z = trans[2]

        elevation = np.arctan2(np.sqrt(x ** 2 + y ** 2), z) / np.pi * 180
        azimuth = np.arctan2(y, x) / np.pi * 180
        distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        if distance > min_vehicle_distance and distance < 50.0:
            coords.append([vehicle.id, x, y, z])


    # Dump the data as txt file
    with open(os.path.join(dump_root_dir, 'vehicles_pos/{:08d}.txt'.format(nowFrame)), 'w') as f:
        for i in range(len(coords)):
            f.write('{},{},{},{}\n'.format(coords[i][0], coords[i][1], coords[i][2], coords[i][3]))

    # Save rgb bev data
    cv2.imwrite(dump_root_dir + 'bev_rgb/{:08d}.png'.format(nowFrame), img_rgb_bev)
    cv2.imwrite(dump_root_dir + 'bev_sem/{:08d}.png'.format(nowFrame), img_sem_bev)


    # Save LiDAR data
    lidar_raw_data = lidar_img.raw_data
    lidar_points = np.frombuffer(lidar_raw_data, dtype=np.dtype('f4'))
    lidar_points = np.reshape(lidar_points, (int(lidar_points.shape[0] / 4), 4))
    lidar_points = np.ascontiguousarray(lidar_points[:, :3])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_points[:, 0:3])
    o3d.io.write_point_cloud(dump_root_dir + 'lidar/{:08d}.pcd'.format(nowFrame), pcd)

    lidar_render = rasterize_lidar(lidar_points, fov=10, distance=200)
    cv2.imwrite(dump_root_dir + 'lidar_render/{:08d}.png'.format(nowFrame), lidar_render)

