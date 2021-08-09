FROM fairembodied/habitat-challenge:testing_2020_habitat_base_docker

RUN /bin/bash -c ". activate habitat; pip install ifcfg tensorboard && pip install torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html"
RUN /bin/bash -c ". activate habitat; pip install scikit-image==0.15.0 && pip install scikit-fmm==2019.1.30 && pip install scikit-learn==0.22.2.post1"
RUN /bin/bash -c ". activate habitat; pip install matplotlib && pip install tensorboard && pip install seaborn==0.9.0 && pip install imageio==2.6.0"
RUN /bin/bash -c ". activate habitat; git clone http://github.com/facebookresearch/habitat-lab.git habitat-lab2 && (cd habitat-lab2 && git checkout 959bd45431edd8024832a877bdc8218015d97a7e) && cp -r habitat-lab2/habitat_baselines habitat-api/."
RUN /bin/bash -c ". activate habitat; conda install -c conda-forge pycocotools"

#ARG INCUBATOR_VER=unknown
ADD semantic.sh semantic.sh
#ADD configs/challenge_objectnav2020.local.rgbd.yaml challenge_objectnav2020.local.rgbd.yaml
ADD pretrained_models pretrained_models
ENV AGENT_EVALUATION_TYPE remote
ENV PYTHONPATH "${PYTHONPATH}:./Neural-SLAM"

ENV TRACK_CONFIG_FILE "/challenge_objectnav2020.local.rgbd.yaml"

CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash semantic.sh"]
