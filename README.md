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

