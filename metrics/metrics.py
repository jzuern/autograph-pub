import torch
import numpy as np
import cv2
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import LineString


#from metrics.apls import compute_apls_metric, make_graphs
#from metrics.geo_topo import Evaluator as GeoTopoEvaluator


def calc_all_metrics(graph_gt, graph_pred, split, imsize=[256, 256]):

    # if len(list(graph_pred.nodes())) > 0:
    #     # plot graph
    #     fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    #     ax[0].set_aspect('equal')
    #     ax[1].set_aspect('equal')
    #     nx.draw(graph_gt, pos=nx.get_node_attributes(graph_gt, 'pos'), ax=ax[0], node_size=10)
    #     nx.draw(graph_pred, pos=nx.get_node_attributes(graph_pred, 'pos'), ax=ax[1], node_size=10)
    #     plt.show()

    iou = calc_iou(graph_gt, graph_pred, imsize)

    # # # Try to calculate APLS metric
    # try:
    #     apls = calc_apls(graph_gt, graph_pred)
    # except Exception as e:
    #     print("APLS metric calculation failed: ", e)  #  File "networkx/classes/graph.py", line 1004, in remove_edge
    #     apls = 0.0
    #
    # # Try to calculate GEO and TOPO metrics
    # try:
    #     graph_gt_ = nx_to_geo_topo_format(graph_gt)
    #     graph_pred_ = nx_to_geo_topo_format(graph_pred)
    #     geo_topo_evaluator = GeoTopoEvaluator(graph_gt_, graph_pred_)
    #     geo_precision, geo_recall, topo_precision, topo_recall = geo_topo_evaluator.topoMetric()
    # except Exception as e:
    #     geo_precision = 0.0
    #     geo_recall = 0.0
    #     topo_precision = 0.0
    #     topo_recall = 0.0
    #     print("Error calculating GEO and TOPO metrics: {}. Continuing".format(e))
    # if np.any(np.isnan([geo_precision, geo_recall, topo_precision, topo_recall])):
    #     geo_precision = 0.0
    #     geo_recall = 0.0
    #     topo_precision = 0.0
    #     topo_recall = 0.0


    metrics_dict = {
        '{}/iou'.format(split): iou,
        # '{}/apls'.format(split): apls,
        # '{}/geo_precision'.format(split): geo_precision,
        # '{}/geo_recall'.format(split): geo_recall,
        # '{}/topo_precision'.format(split): topo_precision,
        # '{}/topo_recall'.format(split): topo_recall,
    }

    return metrics_dict


def calc_apls(graph_gt, graph_pred):

    # Prepare graphs
    for e in graph_gt.edges():
        start = graph_gt.nodes[e[0]]['pos']
        end = graph_gt.nodes[e[1]]['pos']
        lane_segment = LineString([(start[0], start[1]), (end[0], end[1])])
        graph_gt.edges[e]['geometry'] = lane_segment
    for n in graph_gt.nodes():
        pos = graph_gt.nodes[n]['pos']
        graph_gt.nodes[n]['x'] = pos[0]
        graph_gt.nodes[n]['y'] = pos[1]

    for e in graph_pred.edges():
        start = graph_pred.nodes[e[0]]['pos']
        end = graph_pred.nodes[e[1]]['pos']
        lane_segment = LineString([(start[0], start[1]), (end[0], end[1])])
        graph_pred.edges[e]['geometry'] = lane_segment
    for n in graph_pred.nodes():
        pos = graph_pred.nodes[n]['pos']
        graph_pred.nodes[n]['x'] = pos[0]
        graph_pred.nodes[n]['y'] = pos[1]


    # settings
    weight = 'length'
    speed_key = 'inferred_speed_mps'
    travel_time_key = 'travel_time_s'
    linestring_delta = 20.  # Distance between linestring midpoints.
    is_curved_eps = -1.  # can inject nodes everywhere
    max_snap_dist = 5.
    allow_renaming = 1

    G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, control_points_gt, control_points_prop, all_pairs_lengths_gt_native, \
    all_pairs_lengths_prop_native, all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime = \
        make_graphs(graph_gt, graph_pred,
                    weight=weight,
                    speed_key=speed_key,
                    travel_time_key=travel_time_key,
                    linestring_delta=linestring_delta,
                    is_curved_eps=is_curved_eps,
                    max_snap_dist=max_snap_dist,
                    allow_renaming=allow_renaming,
                    verbose=False)

    apls_metric, apls_metric_gt_onto_prop, apls_metric_prop_onto_gt = compute_apls_metric(
        all_pairs_lengths_gt_native, all_pairs_lengths_prop_native,
        all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime,
        control_points_gt, control_points_prop,
        min_path_length=0.1,
        verbose=False, res_dir=None)

    return apls_metric


def render_graph(graph, imsize=[256, 256], width=10):

    """
    Render a graph as an image.
    Args:
        graph:
        imsize:
        width:
    Returns: rendered graph
    """

    im = np.zeros(imsize).astype(np.uint8)

    for e in graph.edges():
        start = graph.nodes[e[0]]['pos']
        end = graph.nodes[e[1]]['pos']
        x1 = int(start[0])
        y1 = int(start[1])
        x2 = int(end[0])
        y2 = int(end[1])
        cv2.line(im, (x1, y1), (x2, y2), 255, width)
    return im


def calc_iou(graph_gt, graph_pred, imsize=[256, 256]):
    """
    Calculate IoU of two graphs.
    :param graph_gt: ground truth graph
    :param graph_pred: predicted graph
    :return: IoU
    """

    render_gt = render_graph(graph_gt, imsize=imsize, width=10)
    render_pred = render_graph(graph_pred, imsize=imsize, width=10)

    # fig, axarr = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    # axarr[0].imshow(render_gt)
    # axarr[1].imshow(render_pred)
    # plt.show()

    # Calculate IoU
    intersection = np.logical_and(render_gt, render_pred)
    union = np.logical_or(render_gt, render_pred)
    iou = np.sum(intersection) / (1e-8 + np.sum(union))

    return iou

def nx_to_geo_topo_format(nx_graph):

    neighbors = {}

    for e in nx_graph.edges():
        x1 = nx_graph.nodes[e[0]]['pos'][0]
        y1 = nx_graph.nodes[e[0]]['pos'][1]
        x2 = nx_graph.nodes[e[1]]['pos'][0]
        y2 = nx_graph.nodes[e[1]]['pos'][1]

        k1 = (int(x1), int(y1))
        k2 = (int(x2), int(y2))

        if k1 not in neighbors:
            neighbors[k1] = []

        if k2 not in neighbors[k1]:
            neighbors[k1].append(k2)

    return neighbors

if __name__ == '__main__':
    graph_gt = nx.read_gpickle('/data/autograph/evaluations/graph_gt.pkl')
    graph_pred = nx.read_gpickle('/data/autograph/evaluations/graph_pred.pkl')

    metrics_dict = calc_all_metrics(graph_gt, graph_pred, split="drive")

    print(metrics_dict)
    print("------------------")


