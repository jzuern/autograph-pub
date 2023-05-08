import pickle
import numpy as np
import pprint
from urbanlanegraph_evaluator.evaluator import GraphEvaluator
from urbanlanegraph_evaluator.utils import adjust_node_positions

city_names = [
    "austin",
    "detroit",
    "miami",
    "paloalto",
    "pittsburgh",
    "washington"
]


def evaluate_successor_lgp(graphs_gt, graphs_pred):

    '''Evaluate the successor graph prediction task.'''

    metric_names = ["TOPO Precision",
                    "TOPO Recall",
                    "GEO Precision",
                    "GEO Recall",
                    "APLS",
                    "SDA20",
                    "SDA50",
                    "Graph IoU"
                    ]

    metrics_all = {}
    split = "eval"

    for city in city_names:
        metrics_all[city] = {}
        metrics_all[city][split] = {}

        for sample_id in graphs_gt[city][split]:
            metrics_all[city][split][sample_id] = {}

            print("Successor-LGP evaluating sample", sample_id)

            if not sample_id in graphs_pred[city][split]:
                print("No prediction for sample", sample_id)
                metrics_sample = {metric_name: 0.0 for metric_name in metric_names}
            else:

                evaluator = GraphEvaluator()

                metrics = evaluator.evaluate_graph(graphs_gt[city][split][sample_id],
                                                   graphs_pred[city][split][sample_id],
                                                   area_size=[256, 256])

                metrics_sample = {
                    "TOPO Precision": metrics['topo_precision'],
                    "TOPO Recall": metrics['topo_recall'],
                    "GEO Precision": metrics['topo_precision'],
                    "GEO Recall": metrics['geo_recall'],
                    "APLS": metrics['apls'],
                    "SDA20": metrics['sda@20'],
                    "SDA50": metrics['sda@50'],
                    "Graph IoU": metrics['iou'],
                }

            metrics_all[city][split][sample_id].update(metrics_sample)

    # Now we average over the samples
    for city in city_names:
        metrics_all[city][split]["avg"] = {}
        for metric_name in metric_names:
            metrics_all[city][split]["avg"][metric_name] = np.nanmean(
                [metrics_all[city][split][sample_id][metric_name] for sample_id in graphs_gt[city][split]])

    # also get the average over all cities
    metrics_all[split] = {}
    metrics_all[split]["avg"] = {}
    for metric_name in metric_names:
        metrics_all[split]["avg"][metric_name] = np.nanmean(
            [metrics_all[city][split]["avg"][metric_name] for city in city_names])

    return metrics_all


def evaluate_full_lgp(graphs_gt, graphs_pred):

    metric_names = ["TOPO Precision",
                    "TOPO Recall",
                    "GEO Precision",
                    "GEO Recall",
                    "APLS",
                    "Graph IoU"
                    ]

    metrics_all = {}
    split = "eval"

    metrics_all[split] = {}

    for city in city_names:
        metrics_all[split][city] = {}

        for sample_id in graphs_gt[city][split]:
            metrics_all[split][city][sample_id] = {}

            print("Full-LGP evaluating sample", sample_id)

            if graphs_pred[city][split][sample_id] is None:
                print("     No prediction for sample", sample_id)
                metrics_sample = {metric_name: 0.0 for metric_name in metric_names}
            else:
                graph_pred = graphs_pred[city][split][sample_id]
                graph_gt = graphs_gt[city][split][sample_id]

                # adjust node positions
                x_offset = float(sample_id.split("_")[2])
                y_offset = float(sample_id.split("_")[3])

                graph_pred = adjust_node_positions(graph_pred, x_offset, y_offset)
                graph_gt = adjust_node_positions(graph_gt, x_offset, y_offset)

                evaluator = GraphEvaluator()
                metrics = evaluator.evaluate_graph(graph_gt,
                                                   graph_pred,
                                                   area_size=[5000, 5000])

                metrics_sample = {
                    "TOPO Precision": metrics['topo_precision'],
                    "TOPO Recall": metrics['topo_recall'],
                    "GEO Precision": metrics['topo_precision'],
                    "GEO Recall": metrics['geo_recall'],
                    "APLS": metrics['apls'],
                    "Graph IoU": metrics['iou'],
                }

            metrics_all[split][city][sample_id].update(metrics_sample)

    # Now we average over the samples
    for city in city_names:
        metrics_all[split][city]["avg"] = {}
        for metric_name in metric_names:
            metrics_all[split][city]["avg"][metric_name] = np.nanmean(
                [metrics_all[split][city][sample_id][metric_name] for sample_id in graphs_gt[city][split]])

    # also get the average over all cities
    metrics_all[split]["avg"] = {}
    for metric_name in metric_names:
        metrics_all[split]["avg"][metric_name] = np.nanmean([metrics_all[split][city]["avg"][metric_name] for city in city_names])

    return metrics_all


def evaluate_planning(graphs_gt, graphs_pred):

    metric_names = ["MMD", "MED", "SR"]
    metrics_all = {}
    split = "eval"

    metrics_all[split] = {}

    for city in city_names:
        metrics_all[split][city] = {}

        for sample_id in graphs_gt[city][split]:
            metrics_all[split][city][sample_id] = {}

            print("Planning evaluating sample", sample_id)

            if graphs_pred[city][split][sample_id] is None:
                print("     No prediction for sample", sample_id)
                metrics_sample = {metric_name: 0.0 for metric_name in metric_names}
            else:
                graph_gt = graphs_gt[city][split][sample_id]
                graph_pred = graphs_pred[city][split][sample_id]

                # adjust node positions
                x_offset = float(sample_id.split("_")[2])
                y_offset = float(sample_id.split("_")[3])

                graph_pred = adjust_node_positions(graph_pred, x_offset, y_offset)
                graph_gt = adjust_node_positions(graph_gt, x_offset, y_offset)

                evaluator = GraphEvaluator()

                paths_gt, paths_pred = evaluator.generate_paths(graph_gt, graph_pred, num_planning_paths=100)
                metrics = evaluator.evaluate_paths(graph_gt, graph_pred, paths_gt, paths_pred)

                metrics_sample = {
                    "MMD": metrics['mmd'],
                    "MED": metrics['med'],
                    "SR": metrics['sr'],
                }

            metrics_all[split][city][sample_id].update(metrics_sample)

    # Now we average over the samples
    for city in city_names:
        metrics_all[split][city]["avg"] = {}
        for metric_name in metric_names:
            metrics_all[split][city]["avg"][metric_name] = np.nanmean(
                [metrics_all[split][city][sample_id][metric_name] for sample_id in graphs_gt[city][split]])

    # also get the average over all cities
    metrics_all[split]["avg"] = {}
    for metric_name in metric_names:
        metrics_all[split]["avg"][metric_name] = np.nanmean(
            [metrics_all[split][city]["avg"][metric_name] for city in city_names])

    return metrics_all


def evaluate(annotation_file, user_submission_file, phase_codename, **kwargs):

    with open(annotation_file, 'rb') as f:
        graphs_gt = pickle.load(f)

    with open(user_submission_file, 'rb') as f:
        graphs_pred = pickle.load(f)

    output = {}
    if phase_codename == "phase_successor_lgp":
        print("%%%%%%%%%%%%%%%%%%%%%\n%%%%%%\tEvaluating for Phase: phase_successor_lgp\n%%%%%%%%%%%%%%%%%%%%%")
        out_dict = evaluate_successor_lgp(graphs_gt, graphs_pred)

        pprint.pprint(out_dict)

        # this goes to the leaderboard (average of all cities
        metrics_successor = out_dict["eval"]["avg"]

        output["result"] = [{"eval_split_succ": metrics_successor}]

        # To display the results in the result file (all cities)
        output["submission_result"] = out_dict

    elif phase_codename == "phase_full_lgp":
        print("%%%%%%%%%%%%%%%%%%%%%\n%%%%%%\tEvaluating for Phase: phase_full_lgp\n%%%%%%%%%%%%%%%%%%%%%")

        out_dict = evaluate_full_lgp(graphs_gt, graphs_pred)

        pprint.pprint(out_dict)

        # the average over all cities for the eval split is this dict entry:
        metrics_full = out_dict["eval"]["avg"]

        output["result"] = [{"eval_split_full": metrics_full}]

        # To display the results in the result file
        output["submission_result"] = output["result"][0]

    elif phase_codename == "phase_planning":
        print("%%%%%%%%%%%%%%%%%%%%%\n%%%%%%\tEvaluating for Phase: phase_planning\n%%%%%%%%%%%%%%%%%%%%%")

        out_dict = evaluate_planning(graphs_gt, graphs_pred)

        pprint.pprint(out_dict)

        # the average over all cities for the eval split is this dict entry:
        metrics_planning = out_dict["eval"]["avg"]

        output["result"] = [{"eval_split_planning": metrics_planning}]

        # To display the results in the result file
        output["submission_result"] = output["result"][0]

    else:
        raise ValueError("Unknown phase codename: {}".format(phase_codename))

    return output


if __name__ == "__main__":

    # Evaluate the submission for each task

    # Task: Successor LGP, Eval Split
    # results_dict = evaluate(annotation_file="annotations_successor_lgp_eval.pickle",
    #                         user_submission_file="succ_lgp_eval_autograph.pickle",
    #                         phase_codename="phase_successor_lgp")

    # # Task: Full LGP, Eval Split
    results_dict = evaluate(annotation_file="annotations_full_lgp_eval.pickle",
                            user_submission_file="/home/zuern/Desktop/autograph/tmp/G_agg/0011_G_agg_cvpr.pickle",
                            phase_codename="phase_full_lgp")
    #
    # # Task: Planning, Eval Split
    # results_dict = evaluate(annotation_file="annotations_full_lgp_eval.pickle",
    #                         user_submission_file="annotations_full_lgp_eval.pickle",
    #                         phase_codename="phase_planning")
