import numpy as np



if __name__ == "__main__":


    Path(viz_output_dir).mkdir(parents=True, exist_ok=True)
    all_scenario_files = sorted(argoverse_scenario_dir.rglob("*.parquet"))
    scenario_file_list = (
        all_scenario_files[:num_scenarios]
        if selection_criteria == SelectionCriteria.FIRST
        else choices(all_scenario_files, k=num_scenarios)
    )  # Ignoring type here because type of "choice" is partially unknown.



    for sf in scenario_file_list:


        track_bounds = None
        for track in scenario.tracks:
            # Get timesteps for which actor data is valid
            actor_timesteps: NDArrayInt = np.array(
                [object_state.timestep for object_state in track.object_states if object_state.timestep <= timestep]
            )
            if actor_timesteps.shape[0] < 1 or actor_timesteps[-1] != timestep:
                continue

            # Get actor trajectory and heading history
            actor_trajectory: NDArrayFloat = np.array(
                [list(object_state.position) for object_state in track.object_states]
            )
            actor_headings: NDArrayFloat = np.array(
                [object_state.heading for object_state in track.object_states]
            )

