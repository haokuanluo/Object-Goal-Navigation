# Parts of the code in this file have been borrowed from:
#    https://github.com/facebookresearch/habitat-api

import numpy as np
import torch
from habitat.config.default import get_config as cfg_env
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.datasets.object_nav.object_nav_dataset import ObjectNavDatasetV1

from .exploration_env import Exploration_Env
from habitat.core.vector_env import VectorEnv
from habitat_baselines.config.default import get_config as cfg_baseline


def make_env_fn(args, config_env, config_baseline, rank):
    #dataset = PointNavDatasetV1(config_env.DATASET)
    config_env.defrost()
    #config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    #print("Loading {}".format(config_env.SIMULATOR.SCENE))
    config_env.freeze()

    env = Exploration_Env(args=args, rank=rank,
                          config_env=config_env, config_baseline=config_baseline, dataset=None
                          )
    env.seed(rank)
    return env


def construct_envs(args):
    env_configs = []
    baseline_configs = []
    args_list = []

    basic_config = cfg_env(config_paths=
                           [args.task_config])
    #basic_config.defrost()
    #basic_config.DATASET.SPLIT = args.split
    #basic_config.freeze()

    scenes = ObjectNavDatasetV1.get_scenes_to_load(basic_config.DATASET)
    print(scenes)

    #if len(scenes) > 0:
    #    assert len(scenes) >= args.num_processes, (
    #        "reduce the number of processes as there "
    #        "aren't enough number of scenes"
    #    )
    #    scene_split_size = int(np.floor(len(scenes) / args.num_processes))

    for i in range(args.num_processes):
        config_env = cfg_env(config_paths=
                             [ args.task_config])
        config_env.defrost()
        #config_env.DATASET.CONTENT_SCENES = scenes[0:1
        #                                            ]
        print(config_env.DATASET.CONTENT_SCENES)
        #if len(scenes) > 0:
        #    config_env.DATASET.CONTENT_SCENES = scenes[
        #                                        i * scene_split_size: (i + 1) * scene_split_size
        #                                        ]

        #if i < args.num_processes_on_first_gpu:
        #    gpu_id = 0
        #else:
        #    gpu_id = int((i - args.num_processes_on_first_gpu)
        #                 // args.num_processes_per_gpu) + args.sim_gpu_id
        #gpu_id = min(torch.cuda.device_count() - 1, gpu_id)
        gpu_id = 0
        config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id
        config_env.ENVIRONMENT.MAX_EPISODE_STEPS = args.max_episode_length
        env_configs.append(config_env)

        #config_baseline = cfg_baseline(config_paths = [args.task_config])
        #baseline_configs.append(config_baseline)

        args_list.append(args)

    envs = VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(args_list, env_configs, range(args.num_processes),
                    range(args.num_processes))
            )
        ),
    )

    return envs
