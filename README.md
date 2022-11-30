# self-supervised-graph


Automatic labeling of lane graphs from vehicle trajectories.


## Run.ai (SCAN cluster)
For more details, see https://wiki.oxfordrobots.com/pages/viewpage.action?spaceKey=PP&title=SCAN+cluster

### build the docker image

```bash
docker build --tag autographv0.1 .
```

### submit interactive job to Run.ai


```bash
# submit interactive job to Run.ai
runai submit test \                                      # runai submits a job called test
     -i docker-image \                                   # -i is the flag for your docker iamge
     -v $/mnt/nfs-1/$USER/self-supervised-graph:/self-supervised-graph/ \       # -v mounts a volume, this volume contains my code and my training data and will be found at location /mydir/mycode
     --working-dir /mydir \                              # set the working directory of the container to /mydir
     --run-as-user \                                     # tells runai to run the container as your user not root --- needed for permissions on the cluster
     -g 0.1 \                                            # how many of a gpu(s) you want
     --interactive \                                     # this is an interactive job                      
     --command -- sleep infinity                         # what command shall be run on entry to the container, here never sleep

# To enter your job and try your code, run:
runai bash test

```



### submit unattended job to Run.ai

```bash
#!/bin/bash -ex
 
PYTHON_SCRIPT=train.py
PATH_TO_CONFIG=config_files/trainSomethingAmazing.yaml
 
export HOME=/home/<username>
 
cd /mydir/my_code
 
pip install -r requirements.txt
 
echo ${PYTHON_SCRIPT}
echo ${PATH_TO_CONFIG}
 
python ${PYTHON_SCRIPT} --c ${PATH_TO_CONFIG}
```


```bash
runai submit test \                                      # runai submits a job called test
     -i docker-image \                                   # -i is the flag for your docker iamge
     -v $/mnt/nfs-1/$USER/mycode:/mydir/my_code/ \       # -v mounts a volume, this volume contains my code and my training data and will be found at location /mydir/mycode
     --working-dir /mydir \                              # set the working directory of the container to /mydir
     --run-as-user \                                     # tells runai to run the container as your user not root --- needed for permissions on the cluster
     -g 1.1 \                                            # how many of a gpu(s) you want                   
     --command /mydir/mycode/startup-scripts/train.sh    # This runs your script to train on entry to the container
```