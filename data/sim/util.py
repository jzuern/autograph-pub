import numpy as np
import cv2
from PIL import Image, ImageDraw
from glob import glob
import os
import matplotlib.pyplot as plt


def rasterize_lidar(pts):

    img_res = [1024, 1024]
    pix_per_m = 20

    # Create empty image
    img = np.zeros((img_res[0], img_res[1]), dtype=np.uint8)

    # only use x,y for now
    pts = pts[:, :2]

    # put points in image space
    pts[:, 0] = pts[:, 0] * pix_per_m + img_res[0] / 2
    pts[:, 1] = pts[:, 1] * pix_per_m + img_res[1] / 2

    # delete all points outside of image
    pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < img_res[0]) & (pts[:, 1] >= 0) & (pts[:, 1] < img_res[1])]

    # render points in image
    for pt in pts:
        img[int(pt[1]), int(pt[0])] = 255

    # plt.imshow(img)
    # plt.show()

    return img



def unproject_image_point(points, intrinsic, distortion):
    f_x = 1000.
    f_y = 1000.
    c_x = 1000.
    c_y = 1000.

    intrinsic = np.array([
        [f_x, 0.0, c_x],
        [0.0, f_y, c_y],
        [0.0, 0.0, 1.0]
    ])

    distortion = np.array([0.0, 0.0, 0.0, 0.0])  # This works!
    distortion = np.array([-0.32, 1.24, 0.0013, 0.0013])  # This doesn't!

    f_x = intrinsic[0, 0]
    f_y = intrinsic[1, 1]
    c_x = intrinsic[0, 2]
    c_y = intrinsic[1, 2]

    # Step 1. Undistort.
    points_undistorted = np.array([])
    if len(points) > 0:
        points_undistorted = cv2.undistortPoints(np.expand_dims(points, axis=1), intrinsic, distortion, P=intrinsic)
        points_undistorted = np.squeeze(points_undistorted, axis=1)

    # Step 2. Reproject.
    result = []
    for idx in range(points_undistorted.shape[0]):
        z = 1 # hardcoded cause it doesnt matter
        x = (points_undistorted[idx, 0] - c_x) / f_x * z
        y = (points_undistorted[idx, 1] - c_y) / f_y * z
        result.append([x, y, z])

    return result



def draw_bbox(carla_img, bboxes):

    img_bgra = np.array(carla_img.raw_data).reshape((carla_img.height, carla_img.width, 4))
    img_rgb = np.zeros((carla_img.height, carla_img.width, 3))

    img_rgb[:, :, 0] = img_bgra[:, :, 2]
    img_rgb[:, :, 1] = img_bgra[:, :, 1]
    img_rgb[:, :, 2] = img_bgra[:, :, 0]
    img_rgb = np.uint8(img_rgb)
    image = Image.fromarray(img_rgb, 'RGB')
    img_draw = ImageDraw.Draw(image)
    for crop in bboxes:
        u1 = int(crop[0, 0])
        v1 = int(crop[0, 1])
        u2 = int(crop[1, 0])
        v2 = int(crop[1, 1])
        crop_bbox = [(u1, v1), (u2, v2)]
        img_draw.rectangle(crop_bbox, outline="red")

    return image


def clean_dumpster(dump_dir):

    print("Deleting files in, ", dump_dir)

    filelist = glob(dump_dir + "*/*.*")

    for f in filelist:
        os.remove(f)


    print("All files deleted")



