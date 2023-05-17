import cv2
import numpy as np
import glob
import os
import pprint


# read image
tiles = glob.glob("/data/lanegraph/urbanlanegraph-dataset-dev/*/tiles/test/*.png")
# tiles = glob.glob("/data/lanegraph/urbanlanegraph-dataset-dev/*/tiles/eval/*.png")

tile_ids = [os.path.basename(t).split(".")[0] for t in tiles]

init_poses = {}


for tile, tile_id in zip(tiles, tile_ids):

    print(tile_id)

    init_poses[tile_id] = []

    img = cv2.imread(tile)
    downsample_factor = 4
    img = cv2.resize(img, (img.shape[1] // downsample_factor, img.shape[0] // downsample_factor))


    # show image
    cv2.imshow('sat_image_viz', img)
    cv2.waitKey(10)

    pose_coordinates = []

    class Visualization:
        def __init__(self):
            self.start = None
            self.end = None

        def cb(self, event, mouseX, mouseY, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.start = np.array([mouseX, mouseY])
            elif event == 3:
                self.end = np.array([mouseX, mouseY])
                cv2.arrowedLine(img, (self.start[0], self.start[1]), (self.end[0], self.end[1]), (0, 0, 255), 2)
                cv2.imshow('sat_image_viz', img)

                angle = np.arctan2(self.end[1] - self.start[1], self.end[0] - self.start[0]) + np.pi / 2
                init_poses[tile_id].append([self.start[0] * downsample_factor,
                                            self.start[1] * downsample_factor,
                                            angle])
                pprint.pprint(init_poses)
                # write to file
                with open("starting_poses_out.json", "w") as f:
                    f.write(str(init_poses))


    viz = Visualization()


    cv2.namedWindow('sat_image_viz')
    cv2.setMouseCallback('sat_image_viz', viz.cb)
    cv2.waitKey(1)
    cv2.waitKey(0)

