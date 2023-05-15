import cv2
import numpy as np
import glob
import os
import pprint


# read image
tiles = glob.glob("/data/lanegraph/urbanlanegraph-dataset-dev/*/tiles/test/*.png")

tile_ids = [os.path.basename(t).split(".")[0] for t in tiles]

init_poses = {}


#tile_ids = ['paloalto_62_35359_38592']



for tile, tile_id in zip(tiles, tile_ids):

    init_poses[tile_id] = []

    img = cv2.imread(tile)
    downsample_factor = 3
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


    viz = Visualization()


    cv2.namedWindow('sat_image_viz')
    cv2.setMouseCallback('sat_image_viz', viz.cb)
    cv2.waitKey(1)
    cv2.waitKey(0)



# init_poses = {
#     "austin_83_34021_46605": np.array([1163, 2982, -2.69]),
#     "austin_40_14021_51605": np.array([1252, 4232,  -np.pi]),
#     "pittsburgh_36_27706_11407": np.array([1789, 2280, 0.4 * np.pi]),
#     'pittsburgh_19_12706_31407': np.array([1789, 2280, 0.4 * np.pi]),
#     "detroit_136_10700_30709": np.array([1335, 3325, 0.2 * np.pi]),
#     "detroit_165_25700_30709": np.array([3032, 1167, - 0.2* np.pi]),
#     "miami_185_41863_18400": np.array([1322, 3287, 0.5 * np.pi]),
#     "miami_194_46863_3400": np.array([3311, 1907, 0.5 * np.pi]),
#     "paloalto_43_25359_23592": np.array([408, 4227, 0 * np.pi]),
#     "paloalto_62_35359_38592": np.array([2980, 3594, 0.5 * np.pi]),
#     "washington_46_36634_59625": np.array([1191, 2467, 0.5 * np.pi]),
#     "washington_55_41634_69625": np.array([871, 1924, 0.5 * np.pi]),
# }

