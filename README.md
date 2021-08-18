# Object Goal Navigation





This is the github repo for object goal navigation. It contains three main components:



### 1. Adapted Neural SLAM



[Learning To Explore Using Active Neural SLAM](https://openreview.net/pdf?id=HklXn1BKDH)<br />
The Neural-SLAM folder contains code that I adapted from the original neural slam github repo. It is modified to work with object-goal navigation goal (go to the chair) from point-goal navigation goal (go to 10 meters forward, 5 meters left) by using a pertained mask-rcnn object recognition model. The modification was made to replicate the object-goal-navigation's results (see below) before the code below is released.

To run:

`sh eval_run.sh `

### 2. Object-Goal-Navigation

[Object Goal Navigation using Goal-Oriented Semantic Exploration](https://arxiv.org/pdf/2007.00643.pdf)<br />

The author's implementation of the above paper. 

To run:

`sh objnav_run.sh`

### 3. semantic

This part of the code is an attempt to train mask-rcnn model by using habitat-api's semantic label api. 

To run:

`sh semantic_run.sh`

### Data:

#### For running Number 1 and Number 2(Modified Neural SLAM and the semantic training models):


The code requires datasets in a `data` folder in the following format (same as habitat-api):
```
Neural-SLAM/
  data/
    scene_datasets/
      gibson/
        Adrian.glb
        Adrian.navmesh
        ...
    datasets/
      pointnav/
        gibson/
          v1/
            train/
            val/
            ...
```
Please download the data using the instructions here: https://github.com/facebookresearch/habitat-api#data


#### For running Number 2(Object Goal Navigation):
#### Downloading scene dataset
- Download the Gibson dataset using the instructions here: https://github.com/facebookresearch/habitat-lab#scenes-datasets (download the 11GB file `gibson_habitat_trainval.zip`)
- Move the Gibson scene dataset or create a symlink at `data/scene_datasets/gibson_semantic`. 

#### Downloading episode dataset
- Download the episode dataset:
```
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1tslnZAkH8m3V5nP8pbtBmaR2XEfr8Rau' -O objectnav_gibson_v1.1.zip
```
- Unzip the dataset into `data/datasets/objectnav/gibson/v1.1/`

#### Setting up datasets
The code requires the datasets in a `data` folder in the following format (same as habitat-lab):
```
Object-Goal-Navigation/
  data/
    scene_datasets/
      gibson_semantic/
        Adrian.glb
        Adrian.navmesh
        ...
    datasets/
      objectnav/
        gibson/
          v1.1/
            train/
            val/
```

