import numpy as np


# PREPROCESSING FILTERING SETTINGS

DBSCAN_MIN_N_SAMPLES = 2  # minimum number of endpoints to be considered a cluster
NUM_ENDPOINTS_MIN = 1  # minimum number of endpoints to be kept
NUM_QUERY_POINTS = 50   # number of query points to be sampled from the rendered trajectory image used to generate successor images
IOU_SIMILARITY_THRESHOLD = 0.7 # minimum iou similarity between two successor trajectories rendered as images to be considered similar


POISSON_DISK_R_MIN = 7 # minimum distance between two randomly sampled points


N_MIN_SUCC_TRAJECTORIES = 2  # minimum number of trajectories that have to be connected to a query position to be considered good sample
FRAC_SUCC_GRAPH_PIXELS = 0.03  # fraction of pixels in the successor visualization that have to be connected to a query position to be considered good sample
# 0.03 is a good value for 256x256 images, for 512x512 images use 0.01


# END PREPROCESSING FILTERING SETTINGS


crop_size = 256  # This is the actual crop size
crop_size_large = 2 * crop_size  # This is twice the actual crop size afterwards




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


# import glob
# import pprint
# tiles_files = glob.glob("/data/lanegraph/urbanlanegraph-dataset-dev/*/tiles/*/*.gpickle")
#
# city_split_coordinates_dict = {}
# for f in tiles_files:
#     city_name = f.split("/")[-4]
#     if city_name not in city_split_coordinates_dict:
#         city_split_coordinates_dict[city_name] = {}
#     split_name = f.split("/")[-2]
#     if split_name not in city_split_coordinates_dict[city_name]:
#         city_split_coordinates_dict[city_name][split_name] = []
#
#     tile_no = int(f.split("/")[-1].split(".")[0].split("_")[1])
#     x_min = int(f.split("/")[-1].split(".")[0].split("_")[2])
#     y_min = int(f.split("/")[-1].split(".")[0].split("_")[3])
#     x_max = x_min + 5000
#     y_max = y_min + 5000
#
#     city_split_coordinates_dict[city_name][split_name].append([x_min, y_min, x_max, y_max])
#
# pprint.pprint(city_split_coordinates_dict)


city_split_coordinates_dict = \
{'Austin': {'eval': [[14021, 51605, 19021, 56605],
                     [34021, 46605, 39021, 51605]],
            'test': [[29021, 46605, 34021, 51605],
                     [14021, 56605, 19021, 61605]],
            'train': [[4021, 41605, 9021, 46605],
                      [14021, 26605, 19021, 31605],
                      [19021, 61605, 24021, 66605],
                      [9021, 21605, 14021, 26605],
                      [44021, 46605, 49021, 51605],
                      [44021, 41605, 49021, 46605],
                      [24021, 41605, 29021, 46605],
                      [24021, 61605, 29021, 66605],
                      [39021, 46605, 44021, 51605],
                      [-978, 56605, 4022, 61605],
                      [19021, 31605, 24021, 36605],
                      [9021, 51605, 14021, 56605],
                      [19021, 26605, 24021, 31605],
                      [4021, 56605, 9021, 61605],
                      [19021, 66605, 24021, 71605],
                      [-978, 46605, 4022, 51605],
                      [4021, 46605, 9021, 51605],
                      [24021, 56605, 29021, 61605],
                      [39021, 36605, 44021, 41605],
                      [14021, 41605, 19021, 46605],
                      [19021, 16605, 24021, 21605],
                      [9021, 36605, 14021, 41605],
                      [39021, 41605, 44021, 46605],
                      [19021, 41605, 24021, 46605],
                      [9021, 41605, 14021, 46605],
                      [14021, 61605, 19021, 66605],
                      [19021, 51605, 24021, 56605],
                      [14021, 46605, 19021, 51605],
                      [4021, 61605, 9021, 66605],
                      [14021, 66605, 19021, 71605],
                      [24021, 66605, 29021, 71605],
                      [19021, 21605, 24021, 26605],
                      [9021, 31605, 14021, 36605],
                      [9021, 46605, 14021, 51605],
                      [24021, 46605, 29021, 51605],
                      [9021, 66605, 14021, 71605],
                      [14021, 16605, 19021, 21605],
                      [14021, 21605, 19021, 26605],
                      [34021, 41605, 39021, 46605],
                      [24021, 51605, 29021, 56605],
                      [-978, 51605, 4022, 56605],
                      [9021, 56605, 14021, 61605],
                      [29021, 41605, 34021, 46605],
                      [4021, 66605, 9021, 71605],
                      [19021, 36605, 24021, 41605],
                      [4021, 51605, 9021, 56605],
                      [14021, 31605, 19021, 36605],
                      [24021, 31605, 29021, 36605],
                      [9021, 61605, 14021, 66605],
                      [19021, 56605, 24021, 61605],
                      [19021, 46605, 24021, 51605],
                      [9021, 26605, 14021, 31605],
                      [9021, 16605, 14021, 21605],
                      [-978, 41605, 4022, 46605],
                      [14021, 36605, 19021, 41605],
                      [24021, 36605, 29021, 41605]]},
 'Detroit': {'eval': [[10700, 35709, 15700, 40709],
                      [25700, 30709, 30700, 35709]],
             'test': [[45700, 25709, 50700, 30709],
                      [10700, 30709, 15700, 35709]],
             'train': [[40700, 25709, 45700, 30709],
                       [15700, 20709, 20700, 25709],
                       [15700, 10709, 20700, 15709],
                       [10700, 10709, 15700, 15709],
                       [10700, 25709, 15700, 30709],
                       [20700, 10709, 25700, 15709],
                       [5700, 20709, 10700, 25709],
                       [25700, 10709, 30700, 15709],
                       [20700, 25709, 25700, 30709],
                       [20700, 30709, 25700, 35709],
                       [15700, 30709, 20700, 35709],
                       [30700, 10709, 35700, 15709],
                       [5700, 25709, 10700, 30709],
                       [15700, 15709, 20700, 20709],
                       [40700, 20709, 45700, 25709],
                       [20700, 20709, 25700, 25709],
                       [25700, 20709, 30700, 25709],
                       [35700, 10709, 40700, 15709],
                       [20700, 5709, 25700, 10709],
                       [20700, 15709, 25700, 20709],
                       [25700, 15709, 30700, 20709],
                       [40700, 10709, 45700, 15709],
                       [35700, 15709, 40700, 20709],
                       [25700, 5709, 30700, 10709],
                       [30700, 20709, 35700, 25709],
                       [15700, 35709, 20700, 40709],
                       [45700, 20709, 50700, 25709],
                       [35700, 25709, 40700, 30709],
                       [15700, 25709, 20700, 30709],
                       [30700, 15709, 35700, 20709],
                       [10700, 15709, 15700, 20709],
                       [40700, 15709, 45700, 20709],
                       [15700, 5709, 20700, 10709],
                       [35700, 30709, 40700, 35709],
                       [30700, 25709, 35700, 30709],
                       [25700, 25709, 30700, 30709],
                       [35700, 20709, 40700, 25709],
                       [30700, 30709, 35700, 35709],
                       [20700, 35709, 25700, 40709],
                       [700, 25709, 5700, 30709]]},
 'Miami': {'eval': [[46863, 3400, 51863, 8400], [41863, 18400, 46863, 23400]],
           'test': [[1863, 43400, 6863, 48400], [21863, 48400, 26863, 53400]],
           'train': [[66863, 8400, 71863, 13400],
                     [36863, 23400, 41863, 28400],
                     [91863, 23400, 96863, 28400],
                     [81863, 8400, 86863, 13400],
                     [26863, 28400, 31863, 33400],
                     [91863, 3400, 96863, 8400],
                     [21863, 13400, 26863, 18400],
                     [46863, 8400, 51863, 13400],
                     [51863, 8400, 56863, 13400],
                     [11863, 23400, 16863, 28400],
                     [6863, 28400, 11863, 33400],
                     [56863, 13400, 61863, 18400],
                     [16863, 18400, 21863, 23400],
                     [56863, 3400, 61863, 8400],
                     [31863, 18400, 36863, 23400],
                     [41863, 3400, 46863, 8400],
                     [31863, 23400, 36863, 28400],
                     [6863, 23400, 11863, 28400],
                     [51863, 3400, 56863, 8400],
                     [1863, 38400, 6863, 43400],
                     [86863, 23400, 91863, 28400],
                     [46863, 28400, 51863, 33400],
                     [96863, 8400, 101863, 13400],
                     [51863, 33400, 56863, 38400],
                     [61863, 8400, 66863, 13400],
                     [1863, 33400, 6863, 38400],
                     [21863, 23400, 26863, 28400],
                     [91863, 8400, 96863, 13400],
                     [31863, 28400, 36863, 33400],
                     [86863, 13400, 91863, 18400],
                     [86863, 3400, 91863, 8400],
                     [1863, 18400, 6863, 23400],
                     [86863, 18400, 91863, 23400],
                     [56863, 28400, 61863, 33400],
                     [21863, 28400, 26863, 33400],
                     [51863, 18400, 56863, 23400],
                     [56863, 23400, 61863, 28400],
                     [71863, 8400, 76863, 13400],
                     [26863, 18400, 31863, 23400],
                     [6863, 33400, 11863, 38400],
                     [56863, 8400, 61863, 13400],
                     [56863, 18400, 61863, 23400],
                     [31863, 33400, 36863, 38400],
                     [16863, 23400, 21863, 28400],
                     [46863, 33400, 51863, 38400],
                     [41863, 33400, 46863, 38400],
                     [1863, 28400, 6863, 33400],
                     [6863, 43400, 11863, 48400],
                     [36863, 18400, 41863, 23400],
                     [51863, 13400, 56863, 18400],
                     [91863, 13400, 96863, 18400],
                     [51863, 23400, 56863, 28400],
                     [46863, 13400, 51863, 18400],
                     [46863, 23400, 51863, 28400],
                     [91863, 18400, 96863, 23400],
                     [51863, 28400, 56863, 33400],
                     [16863, 13400, 21863, 18400],
                     [1863, 48400, 6863, 53400],
                     [41863, 28400, 46863, 33400],
                     [1863, 23400, 6863, 28400],
                     [46863, 38400, 51863, 43400],
                     [21863, 8400, 26863, 13400],
                     [16863, 28400, 21863, 33400],
                     [76863, 8400, 81863, 13400],
                     [41863, 23400, 46863, 28400],
                     [6863, 48400, 11863, 53400],
                     [41863, 13400, 46863, 18400],
                     [21863, 18400, 26863, 23400],
                     [11863, 28400, 16863, 33400],
                     [21863, 33400, 26863, 38400],
                     [26863, 23400, 31863, 28400],
                     [26863, 33400, 31863, 38400],
                     [6863, 18400, 11863, 23400],
                     [46863, 18400, 51863, 23400],
                     [96863, 3400, 101863, 8400],
                     [86863, 8400, 91863, 13400],
                     [81863, 3400, 86863, 8400],
                     [6863, 38400, 11863, 43400],
                     [41863, 8400, 46863, 13400]]},
 'PaloAlto': {'eval': [[35359, 38592, 40359, 43592],
                       [25359, 23592, 30359, 28592]],
              'test': [[30359, 13592, 35359, 18592],
                       [15359, 8592, 20359, 13592]],
              'train': [[20359, 13592, 25359, 18592],
                        [15359, 28592, 20359, 33592],
                        [20359, 28592, 25359, 33592],
                        [10359, 8592, 15359, 13592],
                        [20359, 38592, 25359, 43592],
                        [30359, 38592, 35359, 43592],
                        [359, 28592, 5359, 33592],
                        [35359, 33592, 40359, 38592],
                        [15359, 38592, 20359, 43592],
                        [35359, 18592, 40359, 23592],
                        [30359, 23592, 35359, 28592],
                        [25359, 13592, 30359, 18592],
                        [359, 23592, 5359, 28592],
                        [5359, 23592, 10359, 28592],
                        [15359, 23592, 20359, 28592],
                        [30359, 43592, 35359, 48592],
                        [5359, 18592, 10359, 23592],
                        [40359, 33592, 45359, 38592],
                        [20359, 18592, 25359, 23592],
                        [15359, 13592, 20359, 18592],
                        [25359, 18592, 30359, 23592],
                        [40359, 38592, 45359, 43592],
                        [30359, 28592, 35359, 33592],
                        [15359, 33592, 20359, 38592],
                        [25359, 43592, 30359, 48592],
                        [15359, 18592, 20359, 23592],
                        [25359, 8592, 30359, 13592],
                        [5359, 28592, 10359, 33592],
                        [35359, 13592, 40359, 18592],
                        [10359, 13592, 15359, 18592],
                        [10359, 28592, 15359, 33592],
                        [30359, 18592, 35359, 23592],
                        [10359, 18592, 15359, 23592],
                        [20359, 33592, 25359, 38592],
                        [25359, 33592, 30359, 38592],
                        [20359, 8592, 25359, 13592],
                        [20359, 23592, 25359, 28592],
                        [25359, 28592, 30359, 33592],
                        [35359, 43592, 40359, 48592],
                        [35359, 28592, 40359, 33592],
                        [25359, 38592, 30359, 43592]]},
 'Pittsburgh': {'eval': [[2706, 31407, 7706, 36407],
                         [27706, 11407, 32706, 16407]],
                'test': [[12706, 31407, 17706, 36407],
                         [47706, 26407, 52706, 31407]],
                'train': [[12706, 26407, 17706, 31407],
                          [32706, 21407, 37706, 26407],
                          [42706, 21407, 47706, 26407],
                          [27706, 36407, 32706, 41407],
                          [27706, 21407, 32706, 26407],
                          [47706, 21407, 52706, 26407],
                          [12706, 21407, 17706, 26407],
                          [37706, 36407, 42706, 41407],
                          [52706, 31407, 57706, 36407],
                          [32706, 16407, 37706, 21407],
                          [17706, 21407, 22706, 26407],
                          [47706, 31407, 52706, 36407],
                          [27706, 31407, 32706, 36407],
                          [2706, 26407, 7706, 31407],
                          [27706, 16407, 32706, 21407],
                          [32706, 11407, 37706, 16407],
                          [32706, 31407, 37706, 36407],
                          [22706, 11407, 27706, 16407],
                          [47706, 11407, 52706, 16407],
                          [42706, 16407, 47706, 21407],
                          [7706, 26407, 12706, 31407],
                          [37706, 6407, 42706, 11407],
                          [37706, 21407, 42706, 26407],
                          [17706, 16407, 22706, 21407],
                          [22706, 26407, 27706, 31407],
                          [17706, 31407, 22706, 36407],
                          [42706, 36407, 47706, 41407],
                          [37706, 31407, 42706, 36407],
                          [22706, 16407, 27706, 21407],
                          [32706, 36407, 37706, 41407],
                          [22706, 31407, 27706, 36407],
                          [7706, 31407, 12706, 36407],
                          [42706, 31407, 47706, 36407],
                          [17706, 11407, 22706, 16407],
                          [47706, 36407, 52706, 41407],
                          [52706, 36407, 57706, 41407],
                          [37706, 26407, 42706, 31407],
                          [7706, 16407, 12706, 21407],
                          [37706, 11407, 42706, 16407],
                          [27706, 6407, 32706, 11407],
                          [7706, 21407, 12706, 26407],
                          [47706, 16407, 52706, 21407],
                          [17706, 26407, 22706, 31407],
                          [42706, 26407, 47706, 31407],
                          [32706, 26407, 37706, 31407],
                          [27706, 26407, 32706, 31407],
                          [22706, 6407, 27706, 11407],
                          [12706, 11407, 17706, 16407],
                          [37706, 16407, 42706, 21407],
                          [32706, 6407, 37706, 11407],
                          [12706, 16407, 17706, 21407],
                          [42706, 11407, 47706, 16407],
                          [22706, 21407, 27706, 26407]]},
 'Washington': {'eval': [[36634, 59625, 41634, 64625],
                         [41634, 69625, 46634, 74625]],
                'test': [[36634, 69625, 41634, 74625]],
                'train': [[16634, 49625, 21634, 54625],
                          [46634, 69625, 51634, 74625],
                          [56634, 39625, 61634, 44625],
                          [51634, 49625, 56634, 54625],
                          [46634, 64625, 51634, 69625],
                          [26634, 54625, 31634, 59625],
                          [41634, 54625, 46634, 59625],
                          [6634, 49625, 11634, 54625],
                          [21634, 64625, 26634, 69625],
                          [16634, 39625, 21634, 44625],
                          [21634, 54625, 26634, 59625],
                          [31634, 64625, 36634, 69625],
                          [51634, 44625, 56634, 49625],
                          [26634, 59625, 31634, 64625],
                          [21634, 49625, 26634, 54625],
                          [21634, 44625, 26634, 49625],
                          [11634, 64625, 16634, 69625],
                          [51634, 59625, 56634, 64625],
                          [26634, 49625, 31634, 54625],
                          [21634, 39625, 26634, 44625],
                          [6634, 64625, 11634, 69625],
                          [26634, 64625, 31634, 69625],
                          [16634, 44625, 21634, 49625],
                          [46634, 49625, 51634, 54625],
                          [51634, 54625, 56634, 59625],
                          [16634, 64625, 21634, 69625],
                          [56634, 44625, 61634, 49625],
                          [41634, 59625, 46634, 64625],
                          [46634, 54625, 51634, 59625],
                          [51634, 64625, 56634, 69625],
                          [16634, 54625, 21634, 59625],
                          [11634, 54625, 16634, 59625],
                          [51634, 39625, 56634, 44625],
                          [61634, 44625, 66634, 49625],
                          [11634, 49625, 16634, 54625],
                          [36634, 54625, 41634, 59625],
                          [11634, 44625, 16634, 49625],
                          [11634, 59625, 16634, 64625],
                          [31634, 59625, 36634, 64625],
                          [41634, 49625, 46634, 54625],
                          [6634, 59625, 11634, 64625],
                          [36634, 64625, 41634, 69625],
                          [46634, 59625, 51634, 64625],
                          [31634, 54625, 36634, 59625],
                          [21634, 59625, 26634, 64625],
                          [31634, 49625, 36634, 54625],
                          [16634, 59625, 21634, 64625],
                          [46634, 44625, 51634, 49625],
                          [6634, 54625, 11634, 59625],
                          [41634, 64625, 46634, 69625]]}}
