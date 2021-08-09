import math
import os
import pickle
import sys
import time
import gym
import matplotlib
import numpy as np
import quaternion
import skimage.morphology
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
import random
import copy
if sys.platform == 'darwin':
    matplotlib.use("tkagg")
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import habitat
from habitat import logger

from env.utils.map_builder import MapBuilder
from env.utils.fmm_planner import FMMPlanner
from env.habitat.utils import pose as pu
from env.habitat.utils import visualizations as vu
from env.habitat.utils.supervision import HabitatMaps
import env.utils.semantic_utils as su

from model import get_grid


def _preprocess_depth(depth):
    depth = depth[:, :, 0]*1
    mask2 = depth > 0.99
    depth[mask2] = 0.

    for i in range(depth.shape[1]):
        depth[:,i][depth[:,i] == 0.] = depth[:,i].max()

    mask1 = depth == 0
    depth[mask1] = np.NaN
    depth = depth*1000.
    return depth


class Exploration_Env(habitat.RLEnv):

    def __init__(self, args, rank, config_env, config_baseline, dataset):
        if args.visualize:
            plt.ion()
        if args.print_images or args.visualize:
            self.figure, self.ax = plt.subplots(1,2, figsize=(6*16/9, 6),
                                                facecolor="whitesmoke",
                                                num="Thread {}".format(rank))

        self.args = args
        self.num_actions = 3
        self.dt = 10

        self.rank = rank
        super().__init__(config_env)

        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.stop_next_action = 0
        self.observation_space = gym.spaces.Box(0, 255,
                                                (3, args.frame_height,
                                                    args.frame_width),
                                                dtype='uint8')

        self.mapper = self.build_mapper()

        self.episode_no = 0

        self.res = transforms.Compose([transforms.ToPILImage(),
                    transforms.Resize((args.frame_height, args.frame_width),
                                      interpolation = Image.NEAREST)])
        self.scene_name = None
        self.maps_dict = {}
        self.semantic_model = su.get_maskrcnn_model()
        self.semantic_model.eval()
        self.semantic_pallete = []
        self.device = torch.device("cuda:0")
        self.semantic_model.to(self.device)
        self.goal_believed_to_be_found = False
        self.goal_confidence = 0.0
        self.confidence_threshold = 0.6
        self.eval_stats = {
            'distance_to_goal':0.0,
            'success':0.0,
            'spl':0.0,
            'softspl':0.0,
            'item_found':0.0
        }
        self.eval_stats_by_objective = [ {
            'distance_to_goal':0.0,
            'success':0.0,
            'spl':0.0,
            'softspl':0.0,
            'item_found':0.0
        } for i in range(su.num_object())]
        self.episode_no_by_objective = [0 for i in range(su.num_object())]
        self.success_score_by_objective = [
            [] for i in range(su.num_object())
        ]
        self.false_success_score_by_objective = [
            [] for i in range(su.num_object())
        ]

        self.dump_dir = "{}/dump/{}/{}".format(args.dump_location,
                                                args.exp_name,'confidence.p')

        for i in range(args.semantic_chanel):
            self.semantic_pallete.append((random.random(), random.random(), random.random()))
        self.pit = 0

    def randomize_env(self):
        self._env._episode_iterator._shuffle_iterator()

    def save_trajectory_data(self):
        if "replica" in self.scene_name:
            folder = self.args.save_trajectory_data + "/" + \
                        self.scene_name.split("/")[-3]+"/"
        else:
            folder = self.args.save_trajectory_data + "/" + \
                        self.scene_name.split("/")[-1].split(".")[0]+"/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = folder+str(self.episode_no)+".txt"
        with open(filepath, "w+") as f:
            f.write(self.scene_name+"\n")
            for state in self.trajectory_states:
                f.write(str(state)+"\n")
            f.flush()

    def save_position(self):
        self.agent_state = self._env.sim.get_agent_state()
        self.trajectory_states.append([self.agent_state.position,
                                       self.agent_state.rotation])


    def reset(self):
        print("reset called")
        self.goal_step = 0
        if self.episode_no != 0:
            metrics = self.habitat_env.get_metrics()
            print('this round')
            if metrics['distance_to_goal'] < 1.0:
                metrics['success'] = 1.0
            for m, v in metrics.items():
                print(m,v)
                self.eval_stats[m] += v
                self.eval_stats_by_objective[self.object_goal][m] += v
            if self.goal_believed_to_be_found:
                self.eval_stats['item_found'] += 1.0
                self.eval_stats_by_objective[self.object_goal]['item_found'] += 1.0

                if metrics['success'] == 1.0:
                    self.success_score_by_objective[self.object_goal].append(
                        self.goal_confidence)
                else:
                    self.false_success_score_by_objective[
                        self.object_goal].append(self.goal_confidence)
                with open(self.dump_dir, 'wb') as f:
                    pickle.dump(self.success_score_by_objective, f)
                    pickle.dump(self.false_success_score_by_objective, f)
                    #print('success score')
                    #print(self.success_score_by_objective)

                    #print('fail score')
                    #print(self.false_success_score_by_objective)




            self.episode_no_by_objective[self.object_goal] += 1
            print(" ")
            print(" ")
            print('total episode',self.episode_no)
            for m, v in self.eval_stats.items():
                print(m,v/self.episode_no,end=' ')
            print(' ')
            print('by object')
            for i in range(su.num_object()):
                if self.episode_no_by_objective[i] < 1.0:
                    continue
                print(i,'episode no',self.episode_no_by_objective[i])
                for m, v in self.eval_stats_by_objective[i].items():
                    print(m, v / self.episode_no_by_objective[i],end = ' ')
                print(' ')



        args = self.args
        self.episode_no += 1
        self.goal_believed_to_be_found = False
        self.timestep = 0
        self._previous_action = None
        self.trajectory_states = []
        self.prev_blocked = False

        if args.randomize_env_every > 0:
            if np.mod(self.episode_no, args.randomize_env_every) == 0:
                self.randomize_env()

        # Get Ground Truth Map
        self.explorable_map = None
        self.object_goal = None
        self.goal_confidence = 0.0
        while self.explorable_map is None or self.object_goal is None:
            obs = super().reset()
            goal = su.convert_obj(obs["objectgoal"])
            if goal is None:
                continue
            else:
                self.object_goal = goal
            full_map_size = args.map_size_cm//args.map_resolution
            self.explorable_map = self._get_gt_map(full_map_size)
        print('goal is',self.object_goal)
        self.prev_explored_area = 0.

        if self.args.eval == 1 and self.args.task == 'pointgoal':
            dist, angle = obs["pointgoal"]
            x = int(dist * np.cos(angle) * 20.0)
            y = int(dist * np.sin(angle) * 20.0)
            self.pg_loc = [args.map_size_cm // 2 // args.map_resolution + y,
                           args.map_size_cm // 2 // args.map_resolution + x]
        self.stop_next_action = 0
        # Preprocess observations
        rgb = obs['rgb'].astype(np.uint8)
        self.obs = rgb # For visualization
        if self.args.frame_width != self.args.env_frame_width:
            rgb = np.asarray(self.res(rgb))
        state = rgb.transpose(2, 0, 1)
        depth = _preprocess_depth(obs['depth'])

        # Initialize map and pose
        self.map_size_cm = args.map_size_cm
        self.mapper.reset_map(self.map_size_cm)
        self.curr_loc = [self.map_size_cm/100.0/2.0,
                         self.map_size_cm/100.0/2.0, 0.]
        self.curr_loc_gt = self.curr_loc
        self.last_loc_gt = self.curr_loc_gt
        self.last_loc = self.curr_loc
        self.last_sim_location = self.get_sim_location()

        # Convert pose to cm and degrees for mapper
        mapper_gt_pose = (self.curr_loc_gt[0]*100.0,
                          self.curr_loc_gt[1]*100.0,
                          np.deg2rad(self.curr_loc_gt[2]))

        # Update ground_truth map and explored area
        fp_proj, self.map, fp_explored, self.explored_map = \
            self.mapper.update_map(depth, mapper_gt_pose)

        # Initialize variables
        self.scene_name = self.habitat_env.sim.config.SCENE
        self.visited = np.zeros(self.map.shape)
        self.visited_vis = np.zeros(self.map.shape)
        self.visited_gt = np.zeros(self.map.shape)
        self.collison_map = np.zeros(self.map.shape)
        self.semantic_map = [np.zeros(self.map.shape) for i in range(args.semantic_chanel)]
        self.semantic_vis_map = np.zeros(self.map.shape)

        self.col_width = 1

        # Set info
        self.info = {
            'time': self.timestep,
            'fp_proj': fp_proj,
            'fp_explored': fp_explored,
            'sensor_pose': [0., 0., 0.],
            'pose_err': [0., 0., 0.],
            'map': self.map,
            'explored_map':self.explored_map,
            'rc': self.get_rc()

        }
        self.info['exp_reward'] = 0
        self.info['exp_ratio'] = 0
        self.info['reset'] = 1

        self.save_position()
        #self.object_goal = 3
        self.object_loc = None
        return state, self.info


    def validate_item(self,x1,x2,y1,y2,id):
        myscore = np.sum(self.semantic_map[id][x1:x2+1,y1:y2+1])

        for i in range(self.args.semantic_chanel):
            if i == id:
                continue
            if np.sum(self.semantic_map[i][x1:x2+1,y1:y2+1]) > myscore:
                return False
        return True

    def check_semantic(self,startgt):
        self.semantic_vis_map.fill(0)
        b=np.argwhere(self.semantic_map[self.object_goal]>self.confidence_threshold)
        self.object_loc = None
        mind = 100000

        if len(b) > 0:

            for i in b:
                r,c = i
                if self.semantic_vis_map[r,c] == 0:
                    x1,x2,y1,y2 = su.get_boundary(self.semantic_vis_map,self.semantic_map[self.object_goal],r,c)
                    if (self.validate_item(x1,x2,y1,y2,self.object_goal)):
                        xmid = int((x1+x2)/2)
                        ymid = int((y1+y2)/2)
                        d = pu.get_l2_distance(startgt[0], xmid,
                                               startgt[1], ymid)
                        if d < mind:
                            mind = d
                            self.object_loc = [xmid,ymid]
        if self.object_loc is not None and self.goal_believed_to_be_found == False:
            #print('goal found')
            self.goal_believed_to_be_found = True
            self.goal_confidence = np.amax(self.semantic_map[self.object_goal])








    def update_semantic(self,image,depth,current_pose):
        dic = su.get_semantic_mask(image,self.semantic_model)
        labels = dic['labels']
        masks = dic['masks']
        scores = dic['scores']
        n = masks.shape[0]

        for i in range(n):
            #print(labels[i],su.id2label(labels[i]), self.timestep)

            dep = depth * masks[i,0,:,:] + (10000*(1-masks[i,0,:,:]))
            self.semantic_map[labels[i]] = np.maximum(self.semantic_map[labels[i]], scores[i]*(self.mapper.get_obstacle_map(dep,current_pose) * self.map))

    def step(self, action):
        args = self.args
        self.timestep += 1
        action = action['action']
        # Action remapping
        #action = int(action+0.1)
        if action == 2: # Forward
            action = 1
        elif action == 1: # Right
            action = 3
        elif action == 0: # Left
            action = 2
        if self.stop_next_action == 1:
            action = 0

        self.last_loc = np.copy(self.curr_loc)
        self.last_loc_gt = np.copy(self.curr_loc_gt)
        #if False and self.prev_blocked:
        #    if self._previous_action == 1:
        #        action = 2
        #    else:
        #        action = 1
        self._previous_action = action
        #start = time.perf_counter()
        obs, rew, done, info = super().step(action)
        #print(time.perf_counter()-start)
        #if action == 0:
        #    print(info)
        # Preprocess observations
        rgb = obs['rgb'].astype(np.uint8)
        self.obs = rgb # For visualization
        state2 = rgb.transpose(2, 0, 1)
        if self.args.frame_width != self.args.env_frame_width:
            rgb = np.asarray(self.res(rgb))

        state = rgb.transpose(2, 0, 1)


        depth = _preprocess_depth(obs['depth'])

        # Get base sensor and ground-truth pose
        dx_gt, dy_gt, do_gt = self.get_gt_pose_change()

        dx_base, dy_base, do_base = self.get_base_pose_change(
                                        action, (dx_gt, dy_gt, do_gt))

        self.curr_loc = pu.get_new_pose(self.curr_loc,
                               (dx_base, dy_base, do_base))

        self.curr_loc_gt = pu.get_new_pose(self.curr_loc_gt,
                               (dx_gt, dy_gt, do_gt))

        if not args.noisy_odometry:
            self.curr_loc = self.curr_loc_gt
            dx_base, dy_base, do_base = dx_gt, dy_gt, do_gt

        # Convert pose to cm and degrees for mapper
        mapper_gt_pose = (self.curr_loc_gt[0]*100.0,
                          self.curr_loc_gt[1]*100.0,
                          np.deg2rad(self.curr_loc_gt[2]))


        # Update ground_truth map and explored area
        fp_proj, self.map, fp_explored, self.explored_map = \
                self.mapper.update_map(depth, mapper_gt_pose)

        self.update_semantic(state2, depth,mapper_gt_pose )

        # Update collision map

        if action == 1:
            x1, y1, t1 = self.last_loc
            x2, y2, t2 = self.curr_loc
            if abs(x1 - x2)< 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                self.col_width = min(self.col_width, 9)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold: #Collision
                length = 2
                width = self.col_width
                buf = 3
                self.prev_blocked = True
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05*((i+buf) * np.cos(np.deg2rad(t1)) + \
                                        (j-width//2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05*((i+buf) * np.sin(np.deg2rad(t1)) - \
                                        (j-width//2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r*100/args.map_resolution), \
                               int(c*100/args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                    self.collison_map.shape)
                        self.collison_map[r,c] = 1
            else:
                self.prev_blocked = False

        # Set info
        self.info['time'] = self.timestep
        self.info['fp_proj'] = fp_proj
        self.info['fp_explored']= fp_explored
        self.info['sensor_pose'] = [dx_base, dy_base, do_base]
        self.info['pose_err'] = [dx_gt - dx_base,
                                 dy_gt - dy_base,
                                 do_gt - do_base]
        self.info['reset'] = 0
        self.info['loc_gt'] = self.curr_loc_gt
        self.info['pose_change'] = [dx_gt,dy_gt,do_gt]
        self.info['map'] = self.map
        self.info['explored_map'] = self.explored_map
        self.info['rc'] = self.get_rc()

        if self.timestep%args.num_local_steps==0:
            area, ratio = self.get_global_reward()
            self.info['exp_reward'] = area
            self.info['exp_ratio'] = ratio
        else:
            self.info['exp_reward'] = None
            self.info['exp_ratio'] = None
            #if args.eval == 1:
            self.info['exp_reward'] = 0
            self.info['exp_ratio'] = 0

        self.save_position()

        if self.args.eval == 0:
            if self.info['time'] >= args.max_episode_length:
                done = True
                if self.args.save_trajectory_data != "0":
                    self.save_trajectory_data()
            else:
                done = False
        return state, rew, done, self.info

    def get_reward_range(self):
        # This function is not used, Habitat-RLEnv requires this function
        return (0., 1.0)

    def get_reward(self, observations):
        # This function is not used, Habitat-RLEnv requires this function
        return 0.

    def get_global_reward(self):
        curr_explored = self.explored_map*self.explorable_map
        curr_explored_area = curr_explored.sum()

        reward_scale = self.explorable_map.sum()
        m_reward = (curr_explored_area - self.prev_explored_area)*1.
        m_ratio = m_reward/reward_scale
        m_reward = m_reward * 25./10000. # converting to m^2
        self.prev_explored_area = curr_explored_area

        m_reward *= 0.02 # Reward Scaling

        return m_reward, m_ratio

    def get_done(self, observations):
        #return False
        return self._env._episode_over

    def get_info(self, observations):
        # This function is not used, Habitat-RLEnv requires this function
        info = {}
        return info

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def get_spaces(self):
        return self.observation_space, self.action_space

    def build_mapper(self):
        params = {}
        params['frame_width'] = self.args.env_frame_width
        params['frame_height'] = self.args.env_frame_height
        params['fov'] =  self.args.hfov
        params['resolution'] = self.args.map_resolution
        params['map_size_cm'] = self.args.map_size_cm
        params['agent_min_z'] = 25
        params['agent_max_z'] = 150
        params['agent_height'] = self.args.camera_height * 100
        params['agent_view_angle'] = 0
        params['du_scale'] = self.args.du_scale
        params['vision_range'] = self.args.vision_range
        params['visualize'] = self.args.visualize
        params['obs_threshold'] = self.args.obs_threshold
        self.selem = skimage.morphology.disk(self.args.obstacle_boundary /
                                             self.args.map_resolution)
        mapper = MapBuilder(params)
        return mapper


    def get_sim_location(self):
        agent_state = super().habitat_env.sim.get_agent_state(0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis%(2*np.pi)) < 0.1 or (axis%(2*np.pi)) > 2*np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2*np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o


    def get_gt_pose_change(self):
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do


    def get_base_pose_change(self, action, gt_pose_change):
        dx_gt, dy_gt, do_gt = gt_pose_change

        return dx_gt , dy_gt , do_gt

    def get_rc(self):
        start_x_gt, start_y_gt, start_o_gt = self.curr_loc_gt
        r, c = start_y_gt, start_x_gt
        start_gt = [int(r * 100.0 / self.args.map_resolution),
                    int(c * 100.0 / self.args.map_resolution)]
        start_gt = pu.threshold_poses(start_gt, self.visited_gt.shape)
        return start_gt

    def get_short_term_goal(self, inputs):
        #inputs = inputs[0]
        args = self.args

        # Get Map prediction
        _, _, _, gx1, gx2, gy1, gy2 = inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']
        map_pred = self.map[gx1:gx2, gy1:gy2] # basically ignored input
        exp_pred = self.explored_map[gx1:gx2, gy1:gy2]

        grid = np.rint(map_pred)
        explored = np.rint(exp_pred)

        # Get pose prediction and global policy planning window


        # Get last loc
        last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0/args.map_resolution - gx1),
                      int(c * 100.0/args.map_resolution - gy1)]
        last_start = pu.threshold_poses(last_start, grid.shape)

        # Get curr loc
        start_x,start_y,start_o = self.curr_loc_gt
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0/args.map_resolution - gx1),
                 int(c * 100.0/args.map_resolution - gy1)]
        start = pu.threshold_poses(start, grid.shape)
        #TODO: try reducing this

        self.visited[gx1:gx2, gy1:gy2][start[0]-2:start[0]+3,
                                       start[1]-2:start[1]+3] = 1

        steps = 25
        for i in range(steps):
            x = int(last_start[0] + (start[0] - last_start[0]) * (i+1) / steps)
            y = int(last_start[1] + (start[1] - last_start[1]) * (i+1) / steps)
            self.visited_vis[gx1:gx2, gy1:gy2][x, y] = 1

        # Get last loc ground truth pose
        last_start_x, last_start_y = self.last_loc_gt[0], self.last_loc_gt[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0/args.map_resolution),
                      int(c * 100.0/args.map_resolution)]
        last_start = pu.threshold_poses(last_start, self.visited_gt.shape)

        # Get ground truth pose
        start_x_gt, start_y_gt, start_o_gt = self.curr_loc_gt
        r, c = start_y_gt, start_x_gt
        start_gt = [int(r * 100.0/args.map_resolution),
                    int(c * 100.0/args.map_resolution)]
        start_gt = pu.threshold_poses(start_gt, self.visited_gt.shape)
        #self.visited_gt[start_gt[0], start_gt[1]] = 1

        steps = 25
        for i in range(steps):
            x = int(last_start[0] + (start_gt[0] - last_start[0]) * (i+1) / steps)
            y = int(last_start[1] + (start_gt[1] - last_start[1]) * (i+1) / steps)
            self.visited_gt[x, y] = 1

        self.check_semantic(start_gt)
        # Get goal
        goal = inputs['goal']

        if self.args.eval == 1 and self.args.task == 'pointgoal':
            goal = [self.pg_loc[0] - gx1, self.pg_loc[1] - gy1]
            if pu.get_l2_distance(start[0], goal[0], start[1],
                                  goal[1]) * 5 < 25:
                self.stop_next_action = 1

        if self.object_loc is not None:
            goal = [self.object_loc[0]-gx1,self.object_loc[1]-gy1]
            self.goal_step += 1
        else:
            self.goal_step = 0

        if self.args.eval == 1 and self.object_loc is not None:

            if pu.get_l2_distance(start[0], goal[0], start[1],
                                  goal[1]) * 5 < 25:
                self.stop_next_action = 1
            if self.goal_step > 15:
                self.stop_next_action = 1
        goal = pu.threshold_poses(goal, grid.shape)



        # Get intrinsic reward for global policy
        # Negative reward for exploring explored areas i.e.
        # for choosing explored cell as long-term goal
        self.extrinsic_rew = -pu.get_l2_distance(10, goal[0], 10, goal[1])
        self.intrinsic_rew = -exp_pred[goal[0], goal[1]]

        # Get short-term goal
        stg = self._get_stg(grid, explored, start, np.copy(goal), planning_window)

        # Find GT action
        #if self.args.eval or not self.args.train_local:
        #    gt_action = 0
        #else:
        #gt_action = self._get_gt_action(1 - self.explorable_map, start,
        #                                [int(stg[0]), int(stg[1])],
        #                                planning_window, start_o)
        gt_action = self._get_gt_action(self.collison_map, start,
                                        [int(stg[0]), int(stg[1])],
                                        planning_window, start_o)

        (stg_x, stg_y) = stg
        relative_dist = pu.get_l2_distance(stg_x, start[0], stg_y, start[1])
        relative_dist = relative_dist*5./100.
        angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                stg_y - start[1]))
        angle_agent = (start_o)%360.0
        if angle_agent > 180:
            angle_agent -= 360

        relative_angle = (angle_agent - angle_st_goal)%360.0
        if relative_angle > 180:
            relative_angle -= 360

        def discretize(dist):
            dist_limits = [0.25, 3, 10]
            dist_bin_size = [0.05, 0.25, 1.]
            if dist < dist_limits[0]:
                ddist = int(dist/dist_bin_size[0])
            elif dist < dist_limits[1]:
                ddist = int((dist - dist_limits[0])/dist_bin_size[1]) + \
                    int(dist_limits[0]/dist_bin_size[0])
            elif dist < dist_limits[2]:
                ddist = int((dist - dist_limits[1])/dist_bin_size[2]) + \
                    int(dist_limits[0]/dist_bin_size[0]) + \
                    int((dist_limits[1] - dist_limits[0])/dist_bin_size[1])
            else:
                ddist = int(dist_limits[0]/dist_bin_size[0]) + \
                    int((dist_limits[1] - dist_limits[0])/dist_bin_size[1]) + \
                    int((dist_limits[2] - dist_limits[1])/dist_bin_size[2])
            return ddist

        output = np.zeros((args.goals_size + 1))

        output[0] = int((relative_angle%360.)/5.)
        output[1] = discretize(relative_dist)
        output[2] = gt_action

        self.relative_angle = relative_angle

        if args.visualize or args.print_images:
            dump_dir = "{}/dump/{}/".format(args.dump_location,
                                                args.exp_name)
            ep_dir = '{}/episodes/{}/{}/'.format(
                            dump_dir, self.rank+1, self.episode_no)
            if not os.path.exists(ep_dir):
                os.makedirs(ep_dir)

            if args.vis_type == 1: # Visualize predicted map and pose
                sm = []
                for i in self.semantic_map:
                    nm = copy.deepcopy(i)
                    nm = nm[gx1:gx2,gy1:gy2]
                    nm[nm>=0.2]=1.0
                    nm[nm<0.2]=0
                    sm.append(nm)
                vis_grid = vu.get_colored_map_semantic(np.rint(map_pred),
                                self.collison_map[gx1:gx2, gy1:gy2],
                                sm,
                                self.semantic_pallete,
                                self.visited_vis[gx1:gx2, gy1:gy2],
                                self.visited_gt[gx1:gx2, gy1:gy2],
                                goal,
                                self.explored_map[gx1:gx2, gy1:gy2],
                                self.explorable_map[gx1:gx2, gy1:gy2],
                                self.map[gx1:gx2, gy1:gy2] *
                                    self.explored_map[gx1:gx2, gy1:gy2])
                vis_grid = np.flipud(vis_grid)
                vu.visualize(self.figure, self.ax, self.obs, vis_grid[:,:,::-1],
                            (start_x - gy1*args.map_resolution/100.0,
                             start_y - gx1*args.map_resolution/100.0,
                             start_o),
                            (start_x_gt - gy1*args.map_resolution/100.0,
                             start_y_gt - gx1*args.map_resolution/100.0,
                             start_o_gt),
                            dump_dir, self.rank, self.episode_no,
                            self.timestep, args.visualize,
                            args.print_images, args.vis_type)

            else: # Visualize ground-truth map and pose
                vis_grid = vu.get_colored_map(self.map,
                                self.collison_map,
                                self.visited_gt,
                                self.visited_gt,
                                (goal[0]+gx1, goal[1]+gy1),
                                self.explored_map,
                                self.explorable_map,
                                self.map*self.explored_map)
                vis_grid = np.flipud(vis_grid)
                vu.visualize(self.figure, self.ax, self.obs, vis_grid[:,:,::-1],
                            (start_x_gt, start_y_gt, start_o_gt),
                            (start_x_gt, start_y_gt, start_o_gt),
                            dump_dir, self.rank, self.episode_no,
                            self.timestep, args.visualize,
                            args.print_images, args.vis_type)

        return output

    def _get_gt_map(self, full_map_size):
        self.scene_name = self.habitat_env.sim.config.SCENE
        #logger.error('Computing map for %s', self.scene_name)

        # Get map in habitat simulator coordinates
        self.map_obj = HabitatMaps(self.habitat_env)
        if self.map_obj.size[0] < 1 or self.map_obj.size[1] < 1:
            #logger.error("Invalid map: {}/{}".format(
            #                self.scene_name, self.episode_no))
            return None
        #else:
        #    print("valid map")

        agent_y = self._env.sim.get_agent_state().position.tolist()[1]*100.
        sim_map = self.map_obj.get_map(agent_y, -50., 50.0)

        sim_map[sim_map > 0] = 1.

        # Transform the map to align with the agent
        min_x, min_y = self.map_obj.origin/100.0
        x, y, o = self.get_sim_location()
        x, y = -x - min_x, -y - min_y
        range_x, range_y = self.map_obj.max/100. - self.map_obj.origin/100.

        map_size = sim_map.shape
        scale = 2.
        grid_size = int(scale*max(map_size))
        grid_map = np.zeros((grid_size, grid_size))

        grid_map[(grid_size - map_size[0])//2:
                 (grid_size - map_size[0])//2 + map_size[0],
                 (grid_size - map_size[1])//2:
                 (grid_size - map_size[1])//2 + map_size[1]] = sim_map

        if map_size[0] > map_size[1]:
            st = torch.tensor([[
                    (x - range_x/2.) * 2. / (range_x * scale) \
                             * map_size[1] * 1. / map_size[0],
                    (y - range_y/2.) * 2. / (range_y * scale),
                    180.0 + np.rad2deg(o)
                ]])

        else:
            st = torch.tensor([[
                    (x - range_x/2.) * 2. / (range_x * scale),
                    (y - range_y/2.) * 2. / (range_y * scale) \
                            * map_size[0] * 1. / map_size[1],
                    180.0 + np.rad2deg(o)
                ]])

        rot_mat, trans_mat = get_grid(st, (1, 1,
            grid_size, grid_size), torch.device("cpu"))

        grid_map = torch.from_numpy(grid_map).float()
        grid_map = grid_map.unsqueeze(0).unsqueeze(0)
        translated = F.grid_sample(grid_map, trans_mat)
        rotated = F.grid_sample(translated, rot_mat)

        episode_map = torch.zeros((full_map_size, full_map_size)).float()
        if full_map_size > grid_size:
            episode_map[(full_map_size - grid_size)//2:
                        (full_map_size - grid_size)//2 + grid_size,
                        (full_map_size - grid_size)//2:
                        (full_map_size - grid_size)//2 + grid_size] = \
                                rotated[0,0]
        else:
            episode_map = rotated[0,0,
                              (grid_size - full_map_size)//2:
                              (grid_size - full_map_size)//2 + full_map_size,
                              (grid_size - full_map_size)//2:
                              (grid_size - full_map_size)//2 + full_map_size]



        episode_map = episode_map.numpy()
        episode_map[episode_map > 0] = 1.

        return episode_map


    def _get_stg(self, grid, explored, start, goal, planning_window):

        if self.object_loc is not None:
            return goal

        [gx1, gx2, gy1, gy2] = planning_window

        x1 = min(start[0], goal[0])
        x2 = max(start[0], goal[0])
        y1 = min(start[1], goal[1])
        y2 = max(start[1], goal[1])
        dist = pu.get_l2_distance(goal[0], start[0], goal[1], start[1])
        buf = max(20., dist)
        x1 = max(1, int(x1 - buf))
        x2 = min(grid.shape[0]-1, int(x2 + buf))
        y1 = max(1, int(y1 - buf))
        y2 = min(grid.shape[1]-1, int(y2 + buf))

        rows = explored.sum(1)
        rows[rows>0] = 1
        ex1 = np.argmax(rows)
        ex2 = len(rows) - np.argmax(np.flip(rows))

        cols = explored.sum(0)
        cols[cols>0] = 1
        ey1 = np.argmax(cols)
        ey2 = len(cols) - np.argmax(np.flip(cols))

        ex1 = min(int(start[0]) - 2, ex1)
        ex2 = max(int(start[0]) + 2, ex2)
        ey1 = min(int(start[1]) - 2, ey1)
        ey2 = max(int(start[1]) + 2, ey2)

        x1 = max(x1, ex1)
        x2 = min(x2, ex2)
        y1 = max(y1, ey1)
        y2 = min(y2, ey2)

        traversible = skimage.morphology.binary_dilation(
                        grid[x1:x2, y1:y2],
                        self.selem) != True
        traversible[self.collison_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

        traversible[int(start[0]-x1)-1:int(start[0]-x1)+2,
                    int(start[1]-y1)-1:int(start[1]-y1)+2] = 1

        if goal[0]-2 > x1 and goal[0]+3 < x2\
            and goal[1]-2 > y1 and goal[1]+3 < y2:
            traversible[int(goal[0]-x1)-2:int(goal[0]-x1)+3,
                    int(goal[1]-y1)-2:int(goal[1]-y1)+3] = 1
        else:
            goal[0] = min(max(x1, goal[0]), x2)
            goal[1] = min(max(y1, goal[1]), y2)

        def add_boundary(mat):
            h, w = mat.shape
            new_mat = np.ones((h+2,w+2))
            new_mat[1:h+1,1:w+1] = mat
            return new_mat

        traversible = add_boundary(traversible)

        planner = FMMPlanner(traversible, 360//self.dt)

        reachable = planner.set_goal([goal[1]-y1+1, goal[0]-x1+1])

        stg_x, stg_y = start[0] - x1 + 1, start[1] - y1 + 1
        for i in range(self.args.short_goal_dist):
            stg_x, stg_y, replan = planner.get_short_term_goal([stg_x, stg_y])
        if replan:
            stg_x, stg_y = start[0], start[1]
        else:
            stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y)


    def _get_gt_action(self, grid, start, goal, planning_window, start_o):

        [gx1, gx2, gy1, gy2] = planning_window

        x1 = min(start[0], goal[0])
        x2 = max(start[0], goal[0])
        y1 = min(start[1], goal[1])
        y2 = max(start[1], goal[1])
        dist = pu.get_l2_distance(goal[0], start[0], goal[1], start[1])
        buf = max(5., dist)
        x1 = max(0, int(x1 - buf))
        x2 = min(grid.shape[0], int(x2 + buf))
        y1 = max(0, int(y1 - buf))
        y2 = min(grid.shape[1], int(y2 + buf))

        path_found = False
        goal_r = 0
        while not path_found:
            traversible = skimage.morphology.binary_dilation(
                            grid[gx1:gx2, gy1:gy2][x1:x2, y1:y2],
                            self.selem) != True
            traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1
            traversible[int(start[0]-x1)-1:int(start[0]-x1)+2,
                        int(start[1]-y1)-1:int(start[1]-y1)+2] = 1
            traversible[int(goal[0]-x1)-goal_r:int(goal[0]-x1)+goal_r+1,
                        int(goal[1]-y1)-goal_r:int(goal[1]-y1)+goal_r+1] = 1
            scale = 1
            planner = FMMPlanner(traversible, 360//self.dt, scale)

            reachable = planner.set_goal([goal[1]-y1, goal[0]-x1])

            stg_x_gt, stg_y_gt = start[0] - x1, start[1] - y1
            for i in range(1):
                stg_x_gt, stg_y_gt, replan = \
                        planner.get_short_term_goal([stg_x_gt, stg_y_gt])

            if replan and buf < 100.:
                buf = 2*buf
                x1 = max(0, int(x1 - buf))
                x2 = min(grid.shape[0], int(x2 + buf))
                y1 = max(0, int(y1 - buf))
                y2 = min(grid.shape[1], int(y2 + buf))
            elif replan and goal_r < 50:
                goal_r += 1
            else:
                path_found = True

        stg_x_gt, stg_y_gt = stg_x_gt + x1, stg_y_gt + y1
        if self.object_loc is not None:
            stg_x_gt, stg_y_gt = goal
        angle_st_goal = math.degrees(math.atan2(stg_x_gt - start[0],
                                                stg_y_gt - start[1]))
        angle_agent = (start_o)%360.0
        if angle_agent > 180:
            angle_agent -= 360

        relative_angle = (angle_agent - angle_st_goal)%360.0
        if relative_angle > 180:
            relative_angle -= 360

        if relative_angle > 15.:
            gt_action = 1
        elif relative_angle < -15.:
            gt_action = 0
        else:
            gt_action = 2

        return gt_action

    def toString(self):
        return "it's me"
