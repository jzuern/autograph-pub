
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.distance import cdist


df = 3

def merge_successor_trajectories(q, trajectories_all, sat_image):
    # Get all trajectories that go through query point

    dist_thrsh = 6 * df # in px
    angle_thrsh = np.pi / 4  # in rad

    sat_image_viz = sat_image.copy()

    trajectories_close_q = []
    for t in trajectories_all:
        min_distance_from_q = np.min(np.linalg.norm(t - q, axis=1))
        if min_distance_from_q < dist_thrsh:
            closest_index = np.argmin(np.linalg.norm(t - q, axis=1))
            trajectories_close_q.append(t[closest_index:])

    for t in trajectories_all:
        for i in range(len(t)-1):
            cv2.arrowedLine(sat_image_viz, tuple(t[i].astype(int)), tuple(t[i+1].astype(int)), (0, 0, 255), 1)

    for t in trajectories_close_q:
        for i in range(len(t)-1):
            cv2.arrowedLine(sat_image_viz, tuple(t[i].astype(int)), tuple(t[i+1].astype(int)), (0, 255, 0), 1)


    # then get all trajectories that are close to any of the trajectories
    trajectories_2 = []
    for t0 in trajectories_close_q:
        for t1 in trajectories_all:

            angles0 = np.arctan2(t0[1:, 1] - t0[:-1, 1], t0[1:, 0] - t0[:-1, 0])
            angles0 = np.concatenate([angles0, [angles0[-1]]])

            angles1 = np.arctan2(t1[1:, 1] - t1[:-1, 1], t1[1:, 0] - t1[:-1, 0])
            angles1 = np.concatenate([angles1, [angles1[-1]]])


            # check if t1 is close to t0 at any point
            min_dist = np.amin(cdist(t0, t1), axis=0)
            min_angle = np.amin(cdist(angles0[:, np.newaxis], angles1[:, np.newaxis]), axis=0)

            crit_angle = min_angle < angle_thrsh
            crit_dist = min_dist < dist_thrsh

            crit = crit_angle * crit_dist

            if np.any(crit):
                # if so, merge the trajectories
                # find the first point where the criteria is met
                first_crit = np.where(crit)[0][0]
                trajectories_2.append(t1[first_crit:])

    for t2 in trajectories_2:
        for i in range(len(t2)-1):
            cv2.arrowedLine(sat_image_viz, tuple(t2[i].astype(int)), tuple(t2[i+1].astype(int)), (255, 0, 0), 1)

    return sat_image_viz




if __name__ == '__main__':

    t = np.load("trajectories_0.npy", allow_pickle=True)
    sat_image = cv2.imread("sat_image.png")
    #sat_image = cv2.cvtColor(sat_image, cv2.COLOR_BGR2RGB)

    sat_image = cv2.resize(sat_image, (sat_image.shape[1] * df, sat_image.shape[0] * df))
    t = [t_ * df for t_ in t]
    #sat_image_viz = sat_image.copy()

    q = np.array([100, 100])



    def viz(event, x, y, flags, param):
        global mouseX, mouseY
        if event == cv2.EVENT_LBUTTONDOWN:
            mouseX, mouseY = x, y
            sat_image_viz = merge_successor_trajectories(np.array([mouseX, mouseY]), t, sat_image)
            sat_image_viz = cv2.circle(sat_image_viz, (mouseX, mouseY), 5, (0, 0, 0), -1)

            cv2.imshow('image_0', sat_image_viz)

    print("test")
    cv2.namedWindow('image_0')
    cv2.setMouseCallback('image_0', viz)
    cv2.waitKey(1)


    #cv2.imshow("test", np.random.randint(0, 255, (1, 1, 3)).astype(np.uint8))
    cv2.waitKey(0)