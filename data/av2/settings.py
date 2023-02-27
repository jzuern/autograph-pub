import numpy as np


# PREPROCESSING FILTERING SETTINGS

DBSCAN_MIN_N_SAMPLES = 2  # minimum number of endpoints to be considered a cluster
NUM_ENDPOINTS_MIN = 1  # minimum number of endpoints to be kept
NUM_QUERY_POINTS = 50   # number of query points to be sampled from the rendered trajectory image used to generate successor images
IOU_SIMILARITY_THRESHOLD = 0.7 # minimum iou similarity between two successor trajectories rendered as images to be considered similar


N_MIN_SUCC_TRAJECTORIES = 3  # minimum number of trajectories that have to be connected to a query position to be considered good sample
FRAC_SUCC_GRAPH_PIXELS = 0.03  # fraction of pixels in the successor visualization that have to be connected to a query position to be considered good sample


# END PREPROCESSING FILTERING SETTINGS


def kabsch_umeyama(A, B):

    '''
    Calculate the optimal rigid transformation matrix between 2 sets of N x 3 corresponding points using Kabsch Umeyama algorithm.
    '''
    assert A.shape == B.shape
    n, m = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB

    return R, c, t




def get_transform_params(city_name):
    if "pittsburgh" in city_name:
        print("Using Pittsburgh transform params")
        points_shapefile = coordinates_dict["PIT"]["points_shapefile"]
        points_image = coordinates_dict["PIT"]["points_image"]
    elif "miami" in city_name:
        print("Using Miami transform params")
        points_shapefile = coordinates_dict["MIA"]["points_shapefile"]
        points_image = coordinates_dict["MIA"]["points_image"]
    elif "detroit" in city_name:
        print("Using Detroit transform params")
        points_shapefile = coordinates_dict["DTW"]["points_shapefile"]
        points_image = coordinates_dict["DTW"]["points_image"]
    elif "palo" in city_name:
        print("Using PaloAlto transform params")
        points_shapefile = coordinates_dict["PAO"]["points_shapefile"]
        points_image = coordinates_dict["PAO"]["points_image"]
    elif "washington" in city_name:
        print("Using Washington transform params")
        points_shapefile = coordinates_dict["WDC"]["points_shapefile"]
        points_image = coordinates_dict["WDC"]["points_image"]
    elif "austin" in city_name:
        print("Using Austin transform params")
        points_shapefile = coordinates_dict["ATX"]["points_shapefile"]
        points_image = coordinates_dict["ATX"]["points_image"]
    else:
        raise NotImplementedError("Cant find satellite alignment for ", city_name)

    # Perform the coordinate transformation
    R, c, t = kabsch_umeyama(points_image, points_shapefile)

    transform_params = [R, c, t]

    return transform_params




# Coordinate transformations for Kabsch-Umeyama algorithm
coordinates_dict = {
    'boston': {
        'points_shapefile':
            np.array([
                [749.8, 1585.5, 0],
                [807.3, 497.4, 0],
                [1882.66, 1299.03, 0],
            ]),
        'points_image': np.array([
            [3763., 2628., 0],
            [3985., 9892., 0],
            [11260., 4718., 0],
        ]),
    },
    'PIT': {
        'points_shapefile':
            np.array([
                [909.27, -119.73, 0],
                [4867.8, 2476.4, 0],
                [5667.6, -568.1, 0],
            ]),
        'points_image': np.array([
            [5763, 30661, 0],
            [32138, 13330, 0],
            [37450, 33660, 0],
        ])
    },
    'MIA': {
        'points_shapefile':
            np.array([
                [2074.90, 1764.30, 0],
                [5994.12, -570.43, 0],
                [-3778.6, -261.6, 0],
            ]),
        'points_image': np.array([
            [65969, 10647, 0],
            [92068, 26208, 0],
            [26924, 24162, 0],
        ]),
    },
    'DTW': {
        'points_shapefile':
            np.array([
                [10715.6, 3905.1, 0],
                [10429.7, 5141.65, 0],
                [11752.9, 5233.52, 0],
            ]),
        'points_image': np.array([
            [31901., 25657., 0],
            [29997., 17421., 0],
            [19405. * 2, 8405. * 2, 0],
        ])
    },
    'PAO': {
        'points_shapefile':
            np.array([
                [154.25, -1849.03, 0],
                [972.4, 1468, 0],
                [-3057.08, 3133.05, 0],
            ]),
        'points_image': np.array([
            [17430 * 2, 23024 * 2, 0],
            [20170 * 2, 11967 * 2, 0],
            [6735 * 2, 6423 * 2, 0],
        ])
    },
    'WDC': {
        'points_shapefile':
            np.array([
                [3415.5, 24.32, 0],
                [-1579.6, 1635.7, 0],
                [1704, 8484.4, 0],
            ]),
        'points_image': np.array([
            [40799,    69508, 0],
            [3754 * 2, 29380 * 2, 0],
            [14692 * 2, 6556 * 2, 0],
        ])
    },
    'ATX': {
        'points_shapefile':
            np.array([
                [-734.3, 2558.3, 0],
                [-753.7, -3418.3, 0],
                [2302.6, -1396.2, 0]
            ]),
        'points_image': np.array([
            [7038 * 2, 8747 * 2, 0],
            [6962 * 2., 28660 * 2, 0],
            [17155.5 * 2, 21925 * 2, 0]
        ])
    },
}
