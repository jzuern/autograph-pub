import numpy as np
from pathlib import Path
from av2.datasets.motion_forecasting import scenario_serialization
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from settings import get_transform_params
from PIL import Image
Image.MAX_IMAGE_PIXELS = 2334477275000
import open3d as o3d


if __name__ == "__main__":

    num_scenarios = 10000
    city_name = "austin"

    sat_image = np.asarray(Image.open("/data/lane-segmentation/woven-data/original/Austin_extended.png"))
    roi_xxyy = np.array([15000, 20000, 35000, 40000])
    #roi_xxyy = np.array([0, 40000, 0, 40000])
    sat_image = sat_image[roi_xxyy[2]:roi_xxyy[3], roi_xxyy[0]:roi_xxyy[1], :]

    all_scenario_files = sorted(glob(("/data/argoverse2/motion-forecasting/val/0*/*.parquet")))
    all_scenario_files = all_scenario_files[:num_scenarios]

    [R, c, t] = get_transform_params(city_name)
    print("Generating visualizations for {} scenarios.".format(len(all_scenario_files)))

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(sat_image)

    o3d_lines = []

    for scenario_path in tqdm(all_scenario_files):
        scenario_path = Path(scenario_path)
        scenario_id = scenario_path.stem.split("_")[-1]
        static_map_path = scenario_path.parents[0] / f"log_map_archive_{scenario_id}.json"
        scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)

        if scenario.city_name != city_name:
            continue

        for track in scenario.tracks:
            # Get actor trajectory and heading history
            actor_trajectory = np.array([list(object_state.position) for object_state in track.object_states])
            actor_headings = np.array([object_state.heading for object_state in track.object_states])

            actor_trajectory = actor_trajectory[::5]
            actor_headings = actor_headings[::5]

            if track.object_type == "vehicle":
                color = "b"
            elif track.object_type == "pedestrian":
                color = "r"
            else:
                continue

            # Coordinate transformation
            for i in range(len(actor_trajectory)):
                bb = np.array([actor_trajectory[i, 0], actor_trajectory[i, 1], 0])
                tmp = t + c * R @ bb
                actor_trajectory[i] = tmp[0:2] - roi_xxyy[::2]

            if actor_trajectory[0, 0] < 0 or actor_trajectory[0, 1] < 0 \
                    or actor_trajectory[0, 0] > (roi_xxyy[1]-roi_xxyy[0]) or actor_trajectory[0, 1] > (roi_xxyy[3]-roi_xxyy[2]):
                break

            # plotting
            ax.plot(actor_trajectory[:,0], actor_trajectory[:,1], color=color, linewidth=0.5)
            for i in range(len(actor_trajectory)):
                ax.plot(actor_trajectory[i, 0], actor_trajectory[i, 1], color=color, marker="o", markersize=2)

            # 3d plotting
            lines = [[i, i+1] for i in range(len(actor_trajectory)-1)]
            colors_proj = [[0.5, 0.5, 0.5] for i in range(len(lines))]
            actor_statespace = np.hstack((actor_trajectory, actor_headings.reshape(-1, 1) * 20))
            actor_projection = np.hstack((actor_trajectory, np.zeros_like(actor_headings).reshape(-1, 1)))

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(actor_statespace)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            o3d_lines.append(line_set)

            line_set_proj = o3d.geometry.LineSet()
            line_set_proj.points = o3d.utility.Vector3dVector(actor_projection)
            line_set_proj.lines = o3d.utility.Vector2iVector(lines)
            line_set_proj.colors = o3d.utility.Vector3dVector(colors_proj)
            o3d_lines.append(line_set_proj)


    o3d.visualization.draw_geometries(o3d_lines)

    # axis equal
    ax.set_aspect('equal')
    plt.show()

