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

TESTTILES=(austin_41_14021_56605
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


task(){
   ~/anaconda3/envs/geometric/bin/python driver.py "$1" "$2"
}


# initialize a semaphore with a given number of tokens
open_sem(){
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for((;i>0;i--)); do
        printf %s 000 >&3
    done
}

# run the given command asynchronously and pop/push tokens
run_with_lock(){
    local x
    # this read waits until there is something to read
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
     ( "$@"; )
    # push the return code of the command to the semaphore
    printf '%.3d' $? >&3
    )&
}

open_sem $N_PARALLEL


for TILE in "${TESTTILES[@]}"
do
  ~/anaconda3/envs/geometric/bin/python driver.py $TILE lanegraph &
  sleep 10
done