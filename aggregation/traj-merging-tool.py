
import numpy as np
import cv2
from aggregation.aggregate_av2 import merge_successor_trajectories, get_succ_graph
import time



if __name__ == '__main__':

    t = np.load("trajectories_0.npy", allow_pickle=True)
    sat_image = cv2.imread("sat_image.png")
    sat_image = cv2.resize(sat_image, (sat_image.shape[1], sat_image.shape[0]))


    def viz(event, mouseX, mouseY, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            now = time.time()
            q = np.array([mouseX, mouseY])

            succ_traj,  sat_image_viz = merge_successor_trajectories(q, t, sat_image)
            G, sat_image_viz, sdf, angles, angles_viz = get_succ_graph(q, succ_traj,  sat_image_viz)

            sat_image_viz = cv2.circle(sat_image_viz, (mouseX, mouseY), 5, (0, 0, 0), -1)

            print("Inference time: {:.4f} s".format(time.time() - now))
            cv2.imshow('image_0', sat_image_viz)

    cv2.namedWindow('image_0')
    cv2.setMouseCallback('image_0', viz)
    cv2.waitKey(1)
    cv2.waitKey(0)