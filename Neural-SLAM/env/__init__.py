import torch
import numpy as np
from .habitat import construct_envs


def make_vec_envs(args):
    envs = construct_envs(args)
    envs = VecPyTorch(envs, args.device)
    return envs


# Adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/envs.py#L159
class VecPyTorch():

    def __init__(self, venv, device):
        self.venv = venv
        self.num_envs = venv.num_envs
        #self.observation_space = venv.observation_space
        #self.action_space = venv.action_space
        self.device = device

    def reset(self):
        obs,info = [],[]
        rt = self.venv.reset()
        for i in rt:
            a,b = i
            obs.append(a)
            info.append(b)
        #obs, info = rt
        obs = np.array(obs)
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs, info

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info

    def step(self, actions):
        #print("step called")
        actions = actions.cpu().numpy()
        tmp = self.venv.step(actions)

        #print(tmp)

        obss,rewards,dones,infos = [],[],[],[]
        for tt in tmp:
            obs,reward,done,info = tt
            if not isinstance(obs, np.ndarray):
                obs, info = obs
            obss.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        obss = np.array(obss)
        obss = torch.from_numpy(obss).float().to(self.device)

        #obs, reward, done, info = tmp[0]
        rewards = torch.FloatTensor(rewards)

        #if not isinstance(obs,np.ndarray):
        #    obs,info = obs
        #obs = np.array([obs])
        #obs = torch.from_numpy(obs).float().to(self.device)
        #reward = torch.from_numpy(reward).float()
        #reward = torch.FloatTensor([reward])
        #return obs, reward, [done], [info]
        return obss,rewards,dones,infos

    def get_rewards(self, inputs):
        reward = self.venv.get_rewards(inputs)
        reward = torch.from_numpy(reward).float()
        return reward

    def get_short_term_goal(self, inputs):
        fn = ["get_short_term_goal" for i in range(len(inputs))]
        fa = [{"inputs":k} for k in inputs]
        stg = self.venv.call(function_names = fn,function_args_list = fa)
        stg = np.array(stg)
        #stg = self.venv.get_short_term_goal(inputs)
        stg = torch.from_numpy(stg).float()
        return stg

    def close(self):
        return self.venv.close()
