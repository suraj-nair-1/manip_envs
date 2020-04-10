from manip_envs.tabletop import Tabletop
import argparse
import cv2
import numpy as np
import imageio
import pickle
import copy
import os
'''
Constants
'''
HZ = 10
THRESHOLD = 0.03
TOTAL = 1000

def sample_actions(env, sample_sz):
    act_dim = env.action_space.shape[0]
    acts = np.zeros((act_dim, sample_sz, HZ, 1), dtype=float)
    for a in range(act_dim):
        acts[a] = np.random.uniform(env.action_space.low[a], env.action_space.high[a], (sample_sz, HZ, 1))
    actions = np.concatenate([acts[a] for a in range(act_dim)], 2)
    return actions

def init_env(env, args, savedir):
    PATH = savedir
    #'/iris/u/hjnam/task_exp/our-smm/manip_envs/expert_fewer/'
    obs = env.reset()
    init_block = env.data.qpos[9:12] + np.array([0.0, 0.0, 0.1])
    gripper = env.get_endeff_pos() - np.array([0.0, 0.6, 0.05])
    full_obs = []
    full_acts = []
    i = 0
    while True:
        full_obs.append(obs)
        block = env.data.qpos[9:12] + np.array([0.0, 0.0, 0.1])
        gripper = env.get_endeff_pos() - np.array([0.0, 0.6, 0.05])
        act = np.concatenate([block - gripper, np.array([np.random.uniform(-np.pi, np.pi), -1])], -1)
        full_acts.append(act)
        im = env.save_img(PATH, '', '')
        cv2.imwrite(PATH + 'obs' + str(i)+ '.png', (cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).astype(np.uint8))
        obs, reward, _, _ = env.step(act)
        threshold = np.linalg.norm(block - gripper)
        i += 1
        if threshold < 0.01:
            print(block)
            print(gripper)
            print('===')
            init = {}
            init['full obs'] = np.array(full_obs)
            init['full acts'] = np.array(full_acts)
            pickle.dump(init, open(PATH + 'init.p', 'wb'))
            print('finished env init')
            break


def run_env(env, args, expert_obs, expert_acts, eps, savedir):
    #'/iris/u/hjnam/task_exp/our-smm/manip_envs/expert_fewer/'
    PATH = savedir
    obs = env.reset()
    full_obs = []
    full_acts = [] 
    init = pickle.load(open(PATH + 'init.p', 'rb'))
    init_act = init['full acts']
    for (i, act) in enumerate(init_act):
        obs, _, _, _ = env.step(act)
    init_block  = copy.copy(env.data.qpos[9:12] + np.array([0.0, 0.0, 0.1]))
    block  = copy.copy(env.data.qpos[9:12] + np.array([0.0, 0.0, 0.1]))
    gripper = env.get_endeff_pos() - np.array([0.0, 0.6, 0.05])
    threshold = 0.0
    for i in range(10):
        full_obs.append(obs)
        #if i < 10:
        #    act = np.concatenate([block - gripper, np.array([np.random.uniform(-np.pi, np.pi), -1])], -1)
        #    act[0] *= 100
        #    act[1] *= 100
        #    act[2] *= 100
        #else:
        act = np.random.uniform(low=[-1.0, -1.0, -1.0, -np.pi, -1.0], high=[1.0, 1.0, 1.0, np.pi, 1.0])
        act[4] = -1.0
        full_acts.append(act)
        im = env.save_img(PATH, '', '')
        DIR = PATH + str(eps)
        if not os.path.exists(DIR):
            os.mkdir(DIR)
        cv2.imwrite(DIR + '/obs' + str(i)+ '.png', (cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).astype(np.uint8))
        obs, reward, _, _ = env.step(act)
        block = copy.copy(env.data.qpos[9:12] + np.array([0.0, 0.0, 0.1]))
        gripper = env.get_endeff_pos() - np.array([0.0, 0.6, 0.05])
        x = abs(block[0] - init_block[0])
        y = abs(block[1] - init_block[1])
        new_threshold = np.linalg.norm(block - init_block)
        if new_threshold > threshold:
            threshold = new_threshold
        if i == 9:
            if threshold > 0.005:
                print('found an expert trajectory')
                print('init', init_block)
                print('final', block)
                print('===')
                expert_obs.append(np.array(full_obs))
                expert_acts.append(np.array(full_acts))
                return True
    return False

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("-low", "--low_dim", action='store_true', help="A lower dimensional representation")
    args = parser.parse_args()

    env = Tabletop(low_dim=True, smm=True, exploration_only=True, verbose=0)
    savedir = '/iris/u/hjnam/task_exp/our-smm/manip_envs/expert/'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    init_env(env, args, savedir)
    obs = []
    acts = []
    for eps in range(TOTAL):
        succeed = False
        while not succeed:
            succeed = run_env(env, args, obs, acts, eps, savedir)
        print('finished ', eps)

    expert = {}
    expert['obs'] = np.array(obs)
    expert['acts'] = np.array(acts)
    pickle.dump(expert, open(savedir + 'expert.p', 'wb'))
