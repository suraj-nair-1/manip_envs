from collections import OrderedDict
import numpy as np
from gym.spaces import  Dict , Box
import math
import os
# import torch
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv
from PIL import Image
from pyquaternion import Quaternion
from metaworld.envs.mujoco.utils.rotation import euler2quat
import cv2
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import time


class Tabletop(SawyerXYZEnv):
    def __init__(
            self,
            obj_low=None,
            obj_high=None,
            goal_low=None,
            goal_high=None,
            hand_init_pos=(0, 0.6, 0.2),
            liftThresh=0.04,
            rewMode='orig',
            rotMode='rotz',
            problem="rand",
            door=False, #Add door to the env
            new_door=False,
            tower=False,
            drawer=False,
            stack = False,
            exploration = "hard",
            filepath="test",
            max_path_length=50,
            verbose=1,
            hard=False,
            log_freq=100, # in terms of episode num
            **kwargs
    ):
        self.randomize = False
        self.tower = tower # Makes blocks tall when door is added to the env
        self.new_door = new_door
        self.door = door # if True, add door to the env
        self.hard = hard # if True, blocks are initialized to diff corners
        self.stack = stack # if True, then goal ims are stacked blocks
        self.drawer = drawer
        self.exploration = exploration
        self.max_path_length = max_path_length
        self.cur_path_length = 0
        self.quick_init(locals())
        hand_low=(-0.2, 0.4, 0.0)
        hand_high=(0.2, 0.8, 0.20)
        obj_low=(-0.3, 0.4, 0.1)
        obj_high=(0.3, 0.8, 0.3)
        SawyerXYZEnv.__init__(
            self,
            frame_skip=5,
            action_scale=1./20,
            hand_low=hand_low,
            hand_high=hand_high,
            model_name=self.model_name,
            **kwargs
        )
        if obj_low is None:
            obj_low = self.hand_low

        if goal_low is None:
            goal_low = self.hand_low

        if obj_high is None:
            obj_high = self.hand_high
        
        if goal_high is None:
            goal_high = self.hand_high

        self.liftThresh = liftThresh
        self.rewMode = rewMode
        self.rotMode = rotMode
        self.hand_init_pos = np.array(hand_init_pos)
        self.action_rot_scale = 1./10
        self.action_space = Box(
                np.array([-1, -1, -1, -np.pi, -1]),
                np.array([1, 1, 1, np.pi, 1]),
            )
        self.hand_and_obj_space = Box(
            np.hstack((self.hand_low, obj_low)),
            np.hstack((self.hand_high, obj_high)),
        )

        self.imsize = 64
        self.observation_space = Box(0, 1.0, (self.imsize*self.imsize*3, ))
        self.goal_space = self.observation_space
        
        '''For Logging'''
        self.verbose = verbose
        if self.verbose:
            self.imgs = []
            self.filepath = filepath
            if not os.path.exists(self.filepath):
                os.makedirs(self.filepath)
        self.log_freq = log_freq
        self.epcount = 0 # num episodes so far 
        self.good_qpos = None #self.data.qpos[:7]

    @property
    def model_name(self):
        dirname = os.path.dirname(__file__)
        if self.exploration == "easy":
            filename = os.path.join(dirname, "../assets/sawyer_xyz/sawyer_multiobject.xml") # three easy blocks
        else:
            if self.door:
                filename = os.path.join(dirname, "../assets/sawyer_xyz/sawyer_multiobject_door_v2.xml") # three stacked blocks plus door
                if self.tower:
                    filename = os.path.join(dirname, "../assets/sawyer_xyz/sawyer_multiobject_door_v3.xml") # three tall blocks spread out plus door
            elif self.new_door:
                filename = os.path.join(dirname, "../assets/sawyer_xyz/sawyer_tower_0714.xml") # three tall blocks spread out plus door
            elif self.drawer:
                filename = os.path.join(dirname, "../assets/sawyer_xyz/sawyer_cluttered_drawer.xml") # cluttered drawer
            elif self.stack:
                filename = os.path.join(dirname, "../assets/sawyer_xyz/sawyer_multiobject_stack.xml")
            else:
                filename = os.path.join(dirname, "../assets/sawyer_xyz/sawyer_multiobject_hard.xml") # three blocks but spread out
        return filename

    def change_door_angle(self, angle):
        self.data.qpos[-1] = angle
    
    def step(self, action):
        self.set_xyz_action_rotz(action[:4])
        self.do_simulation([action[-1], -action[-1]])

        ob = None
        ob = self.get_obs()
        reward  = self.compute_reward()
        if self.cur_path_length == self.max_path_length:
            done = True
        else:
            done = False
        
        '''
        For logging
        Render images from every step if saving current episode
        '''
        if self.verbose:
            if self.epcount % self.log_freq == 0:
                im = self.sim.render(self.imsize, self.imsize, camera_name='cam0')
                self.imgs.append(im)

        self.cur_path_length +=1
        if self.door:
            return ob, reward, done, {'green_x': self.data.qpos[9], 
                                      'green_y': self.data.qpos[10], 
                                      'green_z': self.data.qpos[11], 
                                      'pink_x': self.data.qpos[12], 
                                      'pink_y': self.data.qpos[13], 
                                      'pink_z': self.data.qpos[14], 
                                      'blue_x': self.data.qpos[15], 
                                      'blue_y': self.data.qpos[16], 
                                      'blue_z': self.data.qpos[17], 
                                      'door_joint': self.data.qpos[-1],
                                      'hand_x': self.get_endeff_pos()[0],
                                      'hand_y': self.get_endeff_pos()[1],
                                      'hand_z': self.get_endeff_pos()[2],
                                      'dist': - self.compute_reward()}
        elif self.new_door:
            return ob, reward, done, {'green_x': self.data.qpos[9], 
                                      'green_y': self.data.qpos[10], 
                                      'green_z': self.data.qpos[11], 
                                      'pink_x': self.data.qpos[12], 
                                      'pink_y': self.data.qpos[13], 
                                      'pink_z': self.data.qpos[14], 
                                      'blue_x': self.data.qpos[15], 
                                      'blue_y': self.data.qpos[16], 
                                      'blue_z': self.data.qpos[17], 
                                      'door_joint': self.data.qpos[-1],
                                      'hand_x': self.get_endeff_pos()[0],
                                      'hand_y': self.get_endeff_pos()[1],
                                      'hand_z': self.get_endeff_pos()[2],
                                      '3_x': self.data.qpos[18],
                                      '3_y': self.data.qpos[19],
                                      '3_z': self.data.qpos[20],
                                      '4_x': self.data.qpos[21],
                                      '4_y' : self.data.qpos[22],
                                      '4_z': self.data.qpos[23], 
                                      'dist': - self.compute_reward()}
        elif self.drawer:
            return ob, reward, done, {'block0_x': self.data.qpos[9], 
                                      'block0_y': self.data.qpos[10], 
                                      'block0_z': self.data.qpos[11], 
                                      'block1_x': self.data.qpos[12], 
                                      'block1_y': self.data.qpos[13], 
                                      'block1_z': self.data.qpos[14], 
                                      'block2_x': self.data.qpos[15], 
                                      'block2_y': self.data.qpos[16], 
                                      'block2_z': self.data.qpos[17], 
                                      'block3_x': self.data.qpos[18], 
                                      'block3_y': self.data.qpos[19], 
                                      'block3_z': self.data.qpos[20], 
                                      'block4_x': self.data.qpos[21], 
                                      'block4_y': self.data.qpos[22], 
                                      'block4_z': self.data.qpos[23], 
                                      'block5_x': self.data.qpos[24], 
                                      'block5_y': self.data.qpos[25], 
                                      'block5_z': self.data.qpos[26], 
                                      'drawer': self.data.qpos[-1],
                                      'hand_x': self.get_endeff_pos()[0],
                                      'hand_y': self.get_endeff_pos()[1],
                                      'hand_z': self.get_endeff_pos()[2],
                                      'dist': - self.compute_reward()}

        return ob, reward, done, {'green_x': self.data.qpos[9], 
                                  'green_y': self.data.qpos[10], 
                                  'green_z': self.data.qpos[11], 
                                  'pink_x': self.data.qpos[12], 
                                  'pink_y': self.data.qpos[13], 
                                  'pink_z': self.data.qpos[14],
                                  'blue_x': self.data.qpos[15], 
                                  'blue_y': self.data.qpos[16], 
                                  'blue_z': self.data.qpos[17],
                                  'hand_x': self.get_endeff_pos()[0],
                                  'hand_y': self.get_endeff_pos()[1],
                                  'hand_z': self.get_endeff_pos()[2],
                                  'dist': - self.compute_reward()}
   
    def get_obs(self):
        obs = self.sim.render(self.imsize, self.imsize, camera_name="cam0")  / 255.
        return obs
    
    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        if self.door or self.new_door:
            start_id = 9 + self.targetobj*7
        elif self.drawer:
            start_id = 9 + self.targetobj*7
        else:
            start_id = 9 + self.targetobj*3
        if len(pos) < 3:
            qpos[start_id:(start_id+2)] = pos.copy()
            qvel[start_id:(start_id+2)] = 0
        else:
            qpos[start_id:(start_id+3)] = pos.copy()
            qvel[start_id:(start_id+3)] = 0
        self.set_state(qpos, qvel)
          
    def initialize(self):
        self.epcount = -1 # to ensure the first episode starts with 0 idx
        self.cur_path_length = 0

    def reset_model(self, no_reset=False, add_noise=False, just_restore=False):
        ''' For logging '''
        if self.verbose and not just_restore:
            if self.epcount % self.log_freq == 0:
                # save gif of episode
                self.save_gif()
        self.cur_path_length = 0
        if not just_restore:
            self.epcount += 1
        
        if not no_reset: # reset initial block pos
            self._reset_hand()
            for _ in range(100):
                self.do_simulation([0.0, 0.0])
            self.targetobj = np.random.randint(3)
            self.cur_path_length = 0
            obj_num = 3
            if self.new_door:
                obj_num = 5
            elif self.drawer:
                obj_num = 6
            for i in range(obj_num):
                self.targetobj = i
                if self.randomize:
                    init_pos = np.random.uniform(
                    -0.2,
                    0.2,
                    size=(2,),
                )
                elif self.hard:
                    if i == 0:
                        init_pos = [-.2, 0]
                    elif i == 1:
                        init_pos = [-.1, .15]
                    else:
                        init_pos = [ .2, -.1]
                elif self.stack:
                    if i == 0:
                           init_pos = [-.2, -0.15]
                    elif i == 1:
                        init_pos = [-.1, .15]
                    else:
                        init_pos = [ .1, .15]
                else:
                    init_pos = [0.1 * (i-1), 0.15] 
                if self.door:
                    init_pos = [-0.15, 0.75, 0.05 * (i+1)]
                    init_pos[:2] += np.random.normal(loc=0, scale=0.001, size=2)
                    if self.tower:
                        if i == 0:
                            init_pos = [-0.15, 0.8, 0.075]
                        if i == 1:
                            init_pos = [-0.12, 0.6, 0.075]
                        if i == 2:
                            init_pos = [0.25, 0.4, 0.075]
                    object_qvel = self.sim.data.get_joint_qvel('objGeom{}_x'.format(i))
                    object_qvel[:] = 0.
                    self.sim.data.set_joint_qvel('objGeom{}_x'.format(i), object_qvel)
                elif self.new_door:
                    if i == 0:
                        init_pos = [-0.15, 0.8, 0.075]
                    if i == 1:
                        init_pos = [-0.12, 0.6, 0.075]
                    if i == 2:
                        init_pos = [0.25, 0.4, 0.075]
                    if i == 3:
                        init_pos = [0.25, 0.6, 0.075]
                    if i == 4:
                        init_pos = [0.15, 0.6, 0.075]
                    object_qvel = self.sim.data.get_joint_qvel('objGeom{}_x'.format(i))
                    object_qvel[:] = 0.
                    self.sim.data.set_joint_qvel('objGeom{}_x'.format(i), object_qvel)
                
                elif self.drawer:
                    if i == 0:
                        init_pos = [0.35, 0.3, 0.05]
                    if i == 1:
                        init_pos = [-0.12, 0.6, 0.05]
                    if i == 2:
                        init_pos = [0.2, 0.3, 0.05]
                    if i == 3:
                        init_pos = [-0.15, 0.4, 0.05]
                    if i == 4:
                        init_pos = [0.45, 0.6, 0.05]
                    if i == 5:
                        init_pos = [-0.2, 0.7, 0.05]
                    object_qvel = self.sim.data.get_joint_qvel('objGeom{}_x'.format(i))
                    object_qvel[:] = 0.
                    self.sim.data.set_joint_qvel('objGeom{}_x'.format(i), object_qvel)
                if add_noise:
                    init_pos += np.random.uniform(-0.02, 0.02, (2,))
                
                self.obj_init_pos = init_pos
                self._set_obj_xyz(self.obj_init_pos)
                # tower pos needs to be initialized via set_joint_qpos
                if self.door or self.new_door or self.drawer:
                    object_qpos = self.sim.data.get_joint_qpos('objGeom{}_x'.format(i))
                    object_qpos[:3 ] = init_pos
                    object_qpos[3:] = 0.
                    self.sim.data.set_joint_qpos('objGeom{}_x'.format(i), object_qpos)
        if self.door or self.new_door:
            self.change_door_angle(0.0)
        elif self.drawer:
            self.data.qpos[-1] = -0.05
        self.sim.forward()
        o = self.get_obs()
        
        if self.epcount % self.log_freq == 0 and not just_restore:
            self.imgs = []
            im = self.sim.render(self.imsize, self.imsize, camera_name='cam0')
            self.imgs.append(im)
        low_dim_info = {'green_x': self.data.qpos[9], 
                        'green_y': self.data.qpos[10], 
                        'green_z': self.data.qpos[11], 
                        'pink_x': self.data.qpos[12], 
                        'pink_y': self.data.qpos[13], 
                        'pink_z': self.data.qpos[14],
                        'blue_x': self.data.qpos[15], 
                        'blue_y': self.data.qpos[16], 
                        'blue_z': self.data.qpos[17],
                        'hand_x': self.get_endeff_pos()[0],
                        'hand_y': self.get_endeff_pos()[1],
                        'hand_z': self.get_endeff_pos()[2],
                        'dist': - self.compute_reward()}

        if self.door:
            low_dim_info['door_joint'] = self.data.qpos[-1]
        elif self.new_door:
            low_dim_info['3_x'] = self.data.qpos[18]
            low_dim_info['3_y'] = self.data.qpos[19]
            low_dim_info['3_z'] = self.data.qpos[20]
            low_dim_info['4_x'] = self.data.qpos[21]
            low_dim_info['4_y'] = self.data.qpos[22]
            low_dim_info['4_z'] = self.data.qpos[23] 
            low_dim_info['door_joint'] = self.data.qpos[-1]

        elif self.drawer:
            low_dim_info = {'block0_x': self.data.qpos[9], 
                                      'block0_y': self.data.qpos[10], 
                                      'block0_z': self.data.qpos[11], 
                                      'block1_x': self.data.qpos[12], 
                                      'block1_y': self.data.qpos[13], 
                                      'block1_z': self.data.qpos[14], 
                                      'block2_x': self.data.qpos[15], 
                                      'block2_y': self.data.qpos[16], 
                                      'block2_z': self.data.qpos[17], 
                                      'block3_x': self.data.qpos[18], 
                                      'block3_y': self.data.qpos[19], 
                                      'block3_z': self.data.qpos[20], 
                                      'block4_x': self.data.qpos[21], 
                                      'block4_y': self.data.qpos[22], 
                                      'block4_z': self.data.qpos[23], 
                                      'block5_x': self.data.qpos[24], 
                                      'block5_y': self.data.qpos[25], 
                                      'block5_z': self.data.qpos[26], 
                                      'drawer': self.data.qpos[-1],
                                      'hand_x': self.get_endeff_pos()[0],
                                      'hand_y': self.get_endeff_pos()[1],
                                      'hand_z': self.get_endeff_pos()[2],
                                      'dist': - self.compute_reward()}

        return o, low_dim_info 


    def _reset_hand(self, pos=None):
        if self.epcount < 10 and self.cur_path_length == 0:
            self.good_qpos = self.sim.data.qpos[:7].copy()
        self.data.qpos[:7] = self.good_qpos
        if pos is None:
            pos = self.hand_init_pos.copy()
        for _ in range(100):
            self.data.set_mocap_pos('mocap', pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def compute_reward(self):
        return 0.0
    
    def get_goal(self, block, fixed_angle=None):
        ''' Returns a random goal img depending on the desired block/door '''
        goal_pos = None
        angle = 0.
        if self.door:
            # If want to set door as the target, uncomment below
            # hinge between +/- 45 degrees, at least abs > 20 degrees
            while abs(angle) < 0.0872665:# larger than 5 degrees angle
                angle = np.random.uniform(-0.785398, 0.785398)
            if fixed_angle is not None:
                angle = fixed_angle
            print('door angle: {}'.format(angle))
            block_0_pos = self.data.qpos[9:12]
            block_1_pos = self.data.qpos[16:19]
            block_2_pos = self.data.qpos[23:26]
            if self.tower:
                block_0_pos = [-0.15, 0.8, 0.075]
                block_1_pos = [-0.12, 0.6, 0.075]
                block_2_pos = [0.25, 0.4, 0.075]
            gripper_pos = self.sim.data.get_geom_xpos('handle')
            if block is not None:
                block_1_pos[:2] += np.random.uniform(-.05, 0.05, (2,))
                gripper_pos = block_1_pos.copy()
                gripper_pos[:2] += np.random.uniform(-0.02, 0.02, (2,))
                gripper_pos[-1] += np.random.uniform(-0.01, 0.01, (1,))
            self.change_door_angle(angle)
            goal_pos = np.concatenate([gripper_pos, block_0_pos, block_1_pos, block_2_pos])
        elif self.new_door:
            while abs(angle) < 0.0872665:# larger than 5 degrees angle
                angle = np.random.uniform(-0.785398, 0.785398)
            if fixed_angle is not None:
                angle = fixed_angle
            block_0_pos = [-0.15, 0.8, 0.075]
            block_3_pos = [0.25, 0.6, 0.075]
            block_4_pos = [0.15, 0.6, 0.075]
            block_1_pos = [-0.12, 0.6, 0.075]
            block_2_pos = [0.25, 0.4, 0.075]
            gripper_pos = self.sim.data.get_geom_xpos('handle')
            goal_pos = np.concatenate([gripper_pos, block_0_pos, block_1_pos, block_2_pos, block_3_pos, block_4_pos])
            self.change_door_angle(angle)
            
        elif self.drawer:
            # slightly increased the goal range from (0, 0.2) to below
            angle = np.random.uniform(0.05, 0.14)
            angle = -angle
            if fixed_angle is not None:
                angle = fixed_angle
            block_0_pos = [0.35, 0.3, 0.05]
            block_1_pos = [-0.12, 0.6, 0.05]
            block_2_pos = [0.2, 0.3, 0.05]
            block_3_pos = [-0.15, 0.4, 0.05]
            block_4_pos = [0.45, 0.6, 0.05]
            block_5_pos = [-0.2, 0.7, 0.05]
            self.change_door_angle(angle)
            gripper_pos = self.data.get_site_xpos('handleStart')
            goal_pos = np.concatenate([gripper_pos, block_0_pos, block_1_pos, block_2_pos, block_3_pos, block_4_pos, block_5_pos])
            
        elif self.stack:
            # Goals are block1 stacked over block2, block0 untouched
            block_0_pos = [-0.1, 0.15, 0] + np.random.uniform(-0.02, 0.02, (3,)) # green block
            block_0_pos[2] = 0.006
            block_2_pos = self.data.qpos[15:18] + np.random.uniform(-0.04, 0.04, (3,))
            block_2_pos[2] = 0.025
            block_1_pos = block_2_pos.copy() + np.random.uniform(-0.015, 0.015, (3,))
            block_1_pos[2] = 0.025 * 3
            gripper_pos = block_2_pos.copy()
            gripper_pos[2] = -0.02
            gripper_pos[1] += 0.1
            goal_pos = np.concatenate([gripper_pos, block_0_pos, block_1_pos, block_2_pos])
        elif block == 0:
            block_1_pos = [0.0, 0.15, 0] + np.random.uniform(-0.02, 0.02, (3,))# pink block
            block_2_pos = [0.2, 0.15, 0] + np.random.uniform(-0.02, 0.02, (3,)) # blue block
            block_0_pos = np.random.uniform( # green block
                    (-0.2, 0.05, 0.0),  
                    (-0.0, 0.25, 0.20), 
                    size=(3,)) 
            if self.hard:
                block_1_pos = [-.1, .15, 0] + np.random.uniform(-0.02, 0.02, (3,))# pink block 
                block_2_pos = [.2, -.1, 0] + np.random.uniform(-0.02, 0.02, (3,)) # blue block
                block_0_pos = np.random.uniform( # green block
                        (-.22, -0.02, 0.0),  
                        (-.18, 0.02, 0.20), 
                        size=(3,)) 
            block_0_pos += np.random.uniform(-0.02, 0.02, (3,))
            # Make goal pos: Random first block initialization, want gripper hovering over block 
            gripper_pos = block_0_pos.copy()
            gripper_pos += np.random.uniform(-0.02, 0.02, (3,))
            gripper_pos[1] += 0.6 # need to adjust for middle of the table for the gripper being (0.0, 0.6)
            gripper_pos[2] = np.random.uniform(0.0, 0.20)
            goal_pos = np.concatenate((gripper_pos, block_0_pos, block_1_pos, block_2_pos))
            
        elif block == 1:
            block_0_pos = [-0.1, 0.15, 0] + np.random.uniform(-0.02, 0.02, (3,)) # green block
            block_2_pos = [0.1, 0.15, 0] + np.random.uniform(-0.02, 0.02, (3,)) # blue block
            block_1_pos = np.random.uniform( # pink block
                    (-0.1, 0.05, 0),
                    (0.1, 0.25, 0.20),
                    size=(3,))
            if self.hard:
                block_0_pos = [-.2, 0, 0] + np.random.uniform(-0.02, 0.02, (3,))# pink block
                block_2_pos = [.2, -.1, 0] + np.random.uniform(-0.02, 0.02, (3,)) # blue block
                block_1_pos = np.random.uniform( 
                        (-.12, 0.13, 0.0),  
                        (-.08, 0.17, 0.20), 
                        size=(3,)) 
            block_1_pos += np.random.uniform(-0.02, 0.02, (3,))
            gripper_pos = block_1_pos.copy()
            gripper_pos += np.random.uniform(-0.02, 0.02, (3,))
            gripper_pos[1] += 0.6 # need to adjust for middle of the table for the gripper being (0.0, 0.6)
            gripper_pos[2] = np.random.uniform(0.0, 0.20)
            goal_pos = np.concatenate((gripper_pos, block_0_pos, block_1_pos, block_2_pos))
        elif block == 2:
            block_1_pos = [0.0, 0.15, 0] + np.random.uniform(-0.02, 0.02, (3,)) # pink block
            block_0_pos = [-0.1, 0.15, 0] + np.random.uniform(-0.02, 0.02, (3,)) # green block
            block_2_pos = np.random.uniform( # blue block
                    (0.0, 0.1, 0.0),  
                    (0.2, 0.2, 0.20), 
                    size=(3,)) 
            if self.hard:
                block_0_pos = [-.2, 0, 0] + np.random.uniform(-0.02, 0.02, (3,))# pink block
                block_1_pos = [-.1, .15, 0] + np.random.uniform(-0.02, 0.02, (3,))# pink block
                block_2_pos = np.random.uniform( # green block
                        (.18, -0.12, 0.0),  
                        (.22, -0.08, 0.20), 
                        size=(3,)) 
            block_2_pos += np.random.uniform(-0.02, 0.02, (3,))
            gripper_pos = block_2_pos.copy()
            gripper_pos += np.random.uniform(-0.02, 0.02, (3,))
            gripper_pos[1] += 0.6
            gripper_pos[2] = np.random.uniform(0.0, 0.20)
            goal_pos = np.concatenate((gripper_pos, block_0_pos, block_1_pos, block_2_pos))
        
        if self.door or self.new_door or self.drawer:
            goal_img = self.save_goal_img(None, goal_pos, 0, angle=angle)
        else:
            goal_img = self.save_goal_img(None, goal_pos, 0)
        return goal_img
    
    ''' Logging Code: Saves gifs of every log_freq episode, heat maps of gripper and block positions, and plots of gripper-block distances '''

    def take_steps_and_render(self, obs, actions, savename, set_qpos=None):
        '''Returns image after having taken actions from obs.'''
        threshold = 0.05
        repeat = True
        _iters = 0
        if set_qpos is not None:
            self.data.qpos[:] = set_qpos.copy()
        else:
            self.reset_model()
            while repeat:
                obj_num = 3
                if self.new_door:
                    obj_num = 5
                elif self.drawer:
                    obj_num = 6
                for i in range(obj_num):
                    self.targetobj = i
                    if self.door or self.new_door or self.drawer: 
                        self.obj_init_pos = obs[(i+1)*3:((i+1)*3)+3]
                        self._set_obj_xyz(self.obj_init_pos)
                    else:
                        self.obj_init_pos = obs[(i+1)*3:((i+1)*3)+2]
                        self._set_obj_xyz(self.obj_init_pos)
                if not (self.door or self.new_door):
                    error = np.linalg.norm(obs[3:12] - self.data.qpos[9:18])
                    repeat = (error >= threshold)
                    _iters += 1
                else:
                    break
                if _iters > 10:
                    break
            repeat = True
            _iters = 0
            if self.door or self.new_door: 
                self.change_door_angle(obs[-1])
                door_vel = np.array([0.])
                self.sim.data.set_joint_qvel('doorjoint', door_vel)
            elif self.drawer:
                self.data.qpos[-1] = angle
                door_vel = np.array([0.])
                self.sim.data.set_joint_qvel('handle', door_vel)
        self._reset_hand(pos=obs[:3])
        imgs = []
        im = self.sim.render(64, 64, camera_name='cam0')
        imgs.append(im)
        ''' Then take the selected actions '''
        for i in range(actions.shape[0]):
            action = actions[i]
            self.set_xyz_action_rotz(action[:4])
            self.do_simulation([action[-1], -action[-1]])
            im = self.sim.render(64, 64, camera_name='cam0')
            imgs.append(im)
            
        im = self.sim.render(64, 64, camera_name='cam0')
        return im
        
    def _restore(self):
        '''For resetting the env without having to call reset() (i.e. without updating episode count)'''
        self.reset_model(just_restore=True)

    def save_goal_img(self, PATH, goal, eps, actions=None, angle=None):
        '''Returns image with a given goal array of positions for the gripper and blocks.'''
        if self.drawer:
            goal[:3] = self.data.get_site_xpos('handleStart')
        pos = goal[:3]
        for _ in range(100):
            self.data.set_mocap_pos('mocap', pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False

        #  Move blocks to correct positions
        obj_num = 3
        if self.new_door:
            obj_num = 5
        elif self.drawer:
            obj_num = 6
        for i in range(obj_num):
            self.targetobj = i
            init_pos = None
            if self.new_door or self.stack or self.door or self.drawer:
                init_pos = goal[(i+1)*3:((i+1)*3)+3]
                self.obj_init_pos = goal[(i+1)*3:((i+1)*3)+3]
            else:
                self.obj_init_pos = goal[(i+1)*3:((i+1)*3)+2]
            self.obj_init_pos[:2] += np.random.normal(loc=0, scale=0.001, size=2)
                
            self._set_obj_xyz(self.obj_init_pos)
          
            if self.door or self.new_door or self.drawer:
                object_qpos = self.sim.data.get_joint_qpos('objGeom{}_x'.format(i))
                object_qpos[:3] = init_pos
                object_qpos[3:] = 0.
                self.sim.data.set_joint_qpos('objGeom{}_x'.format(i), object_qpos)
                object_qvel = self.sim.data.get_joint_qvel('objGeom{}_x'.format(i))
                object_qvel[:] = 0.
                self.sim.data.set_joint_qvel('objGeom{}_x'.format(i), object_qvel)
            self.sim.forward()
        
        if angle is not None:
            self.change_door_angle(angle)
        im = self.sim.render(64, 64, camera_name='cam0') #cam0')
        return im

    
    def save_gif(self):
        ''' Saves the gif of an episode '''
        with imageio.get_writer(
                self.filepath + '/Eps' + str(self.epcount) + '.gif', mode='I') as writer:
            for i in range(self.max_path_length + 1):
                writer.append_data(self.imgs[i])

                
