export PYTHONPATH=$PYTHONPATH:~/self-supervised-graph/
export N_PARALLEL=6

EVALTILES=(
  austin_40_14021_51605
  austin_83_34021_46605
  detroit_136_10700_30709
  detroit_165_25700_30709
  miami_185_41863_18400
  miami_194_46863_3400
  paloalto_43_25359_23592
  paloalto_62_35359_38592
  pittsburgh_36_27706_11407
  pittsburgh_5_2706_31407
  washington_46_36634_59625
  washington_55_41634_69625
)

TESTTILES=(
austin_41_14021_56605
  austin_72_29021_46605
  detroit_135_10700_30709
  detroit_204_45700_25709
  miami_143_21863_48400
  miami_94_1863_43400
  paloalto_24_15359_8592
  paloalto_49_30359_13592
  pittsburgh_19_12706_31407
  pittsburgh_67_47706_26407
  washington_48_36634_69625
)

FREIBURGTILES=(
freiburg_0_0
freiburg_0_10661
freiburg_0_2665
freiburg_0_5330
freiburg_0_7995
freiburg_12086_0
freiburg_12086_10661
freiburg_12086_2665
freiburg_12086_5330
freiburg_12086_7995
freiburg_16115_0
freiburg_16115_10661
freiburg_16115_2665
freiburg_16115_5330
freiburg_16115_7995
freiburg_4028_0
freiburg_4028_10661
freiburg_4028_2665
freiburg_4028_5330
freiburg_4028_7995
freiburg_8057_0
freiburg_8057_10661
freiburg_8057_2665
freiburg_8057_5330
freiburg_8057_7995
)


N_PARALLEL=6


# Function to execute commands in parallel
execute_commands() {
    for TILE in "${FREIBURGTILES[@]}"; do
        # Execute command in the background
        ~/anaconda3/envs/geometric/bin/python driver.py $TILE tracklets &

        # Store the process ID
        pids+=($!)

        # Limit the number of parallel processes
        if [ ${#pids[@]} -eq $N_PARALLEL ]; then
            # Wait for any of the processes to finish
            wait -n

            # Remove the finished process from the list
            pids=("${pids[@]:1}")
        fi
    done

    # Wait for all remaining processes to finish
    wait
}

# Execute the commands
execute_commands

