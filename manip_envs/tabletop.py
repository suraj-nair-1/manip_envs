from collections import OrderedDict
import numpy as np
from gym.spaces import  Dict , Box
import math
import os
import torch
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
            random_init=False,
            tasks = [{'goal': np.array([0.1, 0.8, 0.2]),  
                      'obj_init_pos':np.array([0, 0.6, 0.02]), 
                      'obj_init_angle': 0.3}], 
            goal_low=None,
            goal_high=None,
            hand_init_pos=(0, 0.6, 0.2),
            liftThresh=0.04,
            rewMode='orig',
            rotMode='rotz',
            problem="rand",
            door=True, #Add door to the env
            tower=True,
            drawer=False,
            stack = False,
            exploration = "hard",
            low_dim=False, #True,
            filepath="test",
            max_path_length=50,
            verbose=1,
            double_target=False,
            hard=False,
            log_freq=100, # in terms of episode num
            smm=True, #False,
            exploration_only=False,
            **kwargs
    ):
        self.randomize = False
        self.smm = smm
        self.double_target = double_target
        self.tower = tower # Makes blocks tall when door is added to the env
        self.debug_count = 0
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

        self.imsize= 64
        self.lowdim = low_dim
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

        if self.lowdim:
            if self.smm:
                # Observations are : gripper xyz, block0 xyz, block1 xyz, block2 xyz, 
                    # block0 dxdydz, block1 dxdydz, block2 dxdydz, gripper left and right, gripper z joint
                # Total dim: 3 + 9 + 9 + 2 + 1 = 24
                self.observation_space = Box(0, 1.0, (24,))
            else: 
                self.observation_space = Dict({
                    'state_observation':Box(0, 1.0, (24,))
                })
        else:
            if self.smm:
                self.imsize = 64 #64
                self.observation_space = Box(0, 1.0, (self.imsize*self.imsize*3, ))
            else: 
                self.observation_space = Dict({
                  'image_observation':Box(0, 1.0, (self.imsize*self.imsize*3, )),
                  'image_desired_goal':Box(0, 1.0, (self.imsize*self.imsize*3, )),
                  'image_achieved_goal':Box(0, 1.0, (self.imsize*self.imsize*3, ))
                })
            
        self.goal_space = self.observation_space
        
        # Extra
        self.exploration_only = exploration_only # compute_reward returns 0.0
        
        '''For Logging'''
        self.verbose = verbose
        if self.verbose:
            self.imgs = []
            self.filepath = filepath
            if not os.path.exists(self.filepath):
                os.makedirs(self.filepath)
        self.log_freq = log_freq
        self.epcount = 0 # num episodes so far (start from 1 for logging simplicity)
        self.hand_memory = []
        self.obj_memory0 = []
        self.obj_memory1 = []
        self.obj_memory2 = []
        self.interaction = False # True if you want to log hand-block distances, false otherwise
        if self.interaction:
            self.block0_interaction = []
            self.block1_interaction = []
            self.block2_interaction = []

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
                if self.double_target:
                    filename = os.path.join(dirname, "../assets/sawyer_xyz/sawyer_multiobject_door_block.xml")
            elif self.drawer:
                filename = os.path.join(dirname, "../assets/sawyer_xyz/sawyer_cluttered_drawer.xml") # cluttered drawer
            elif self.stack:
                filename = os.path.join(dirname, "../assets/sawyer_xyz/sawyer_multiobject_stack.xml")
            else:
                filename = os.path.join(dirname, "../assets/sawyer_xyz/sawyer_multiobject_hard.xml") # three blocks but spread out
        return filename

    def change_door_angle(self, angle):
        # old_jt = self.data.qpos.copy()[-1]
        self.data.qpos[-1] = angle
        # print("Door joint before: {} | now: {}".format(old_jt, self.data.qpos[-1]))
    
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
        Save object interaction from the past ~self.log_freq episodes
        Render images from every step if current episode = log_freq_episode
        '''
        if self.verbose:
            hand = self.get_endeff_pos()[:3].copy()
            # get_geom_xpos('objGeom0') = same as get_site_xpos('obj0')
            block0 = self.data.get_geom_xpos('objGeom0')[:3].copy()
            block1 = self.data.get_geom_xpos('objGeom1')[:3].copy()
            block2 = self.data.get_geom_xpos('objGeom2')[:3].copy()
            self.hand_memory.append(hand)
            self.obj_memory0.append(block0)
            self.obj_memory1.append(block1)
            self.obj_memory2.append(block2)
                
            if self.epcount % self.log_freq == 0:
                im = self.sim.render(64, 64, camera_name='cam0')
                self.imgs.append(im)
                # cv2.imwrite(self.filepath + '/obs'+str(self.cur_path_length)+'.png', (cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).astype(np.uint8))
                
            if self.interaction:
                dist0 = np.linalg.norm(block0 - hand)
                dist1 = np.linalg.norm(block1 - hand)
                dist2 = np.linalg.norm(block2 - hand)
                self.block0_interaction.append(dist0)
                self.block1_interaction.append(dist1)
                self.block2_interaction.append(dist2)
        self.cur_path_length +=1
        if self.door:
            return ob, reward, done, {'green_x': self.data.get_site_xpos('obj0')[0], 
                                      'green_y': self.data.get_site_xpos('obj0')[1], 
                                      'green_z': self.data.get_site_xpos('obj0')[2], 
                                      'pink_x': self.data.get_site_xpos('obj1')[0], 
                                      'pink_y': self.data.get_site_xpos('obj1')[1], 
                                      'pink_z': self.data.get_site_xpos('obj1')[2], 
                                      'blue_x': self.data.get_site_xpos('obj2')[0], 
                                      'blue_y': self.data.get_site_xpos('obj2')[1], 
                                      'blue_z': self.data.get_site_xpos('obj2')[2], 
                                      'door_joint': self.data.qpos[-1],
                                      'hand_x': self.get_endeff_pos()[0],
                                      'hand_y': self.get_endeff_pos()[1],
                                      'hand_z': self.get_endeff_pos()[2],
                                      'dist': - self.compute_reward()}
        elif self.drawer:
            return ob, reward, done, {'block0_x': self.data.get_site_xpos('obj0')[0], 
                                      'block0_y': self.data.get_site_xpos('obj0')[1], 
                                      'block0_z': self.data.get_site_xpos('obj0')[2], 
                                      'block1_x': self.data.get_site_xpos('obj1')[0], 
                                      'block1_y': self.data.get_site_xpos('obj1')[1], 
                                      'block1_z': self.data.get_site_xpos('obj1')[2], 
                                      'block2_x': self.data.get_site_xpos('obj2')[0], 
                                      'block2_y': self.data.get_site_xpos('obj2')[1], 
                                      'block2_z': self.data.get_site_xpos('obj2')[2], 
                                      'block3_x': self.data.get_site_xpos('obj3')[0], 
                                      'block3_y': self.data.get_site_xpos('obj3')[1], 
                                      'block3_z': self.data.get_site_xpos('obj3')[2], 
                                      'block4_x': self.data.get_site_xpos('obj4')[0], 
                                      'block4_y': self.data.get_site_xpos('obj4')[1], 
                                      'block4_z': self.data.get_site_xpos('obj4')[2], 
                                      'block5_x': self.data.get_site_xpos('obj5')[0], 
                                      'block5_y': self.data.get_site_xpos('obj5')[1], 
                                      'block5_z': self.data.get_site_xpos('obj5')[2], 
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
   
    def get_obs(self, goal=False):
        if self.lowdim:
            if goal:
                return self.goalst
            gpos = self.get_endeff_pos()
            gquat = self.data.mocap_quat[0]
            gleft = self.data.qpos[7]
            gright = self.data.qpos[8]
            zjoint = self.data.qpos[6]
            objpos = self.data.qpos[9:18]
            objvel = self.data.qvel[9:18]
            obs = np.concatenate([gpos, objpos, objvel, np.array([gleft]), np.array([gright]), np.array([zjoint])])
            
            obs = {'state_observation' :obs}
            
            # for smm
            if self.smm:
                return obs['state_observation']
        else:
            if goal:
                return self.goalim
            im = self.render() #.flatten()
            obs = {'image_observation' :im}
            obs['image_desired_goal'] = self.goalim
            obs['image_achieved_goal'] = im
            
            if self.smm:
                return obs['image_observation']
            
        return obs

    def _get_info(self):
        pass
    
    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        if self.door:
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

    def render(self, mode=""):
        i =  self.sim.render(self.imsize, self.imsize, camera_name="cam0")  / 255. #cam0
        #i = np.swapaxes(i, 0, 2)
        return i
      
    def get_goal(self):
        goal  = {}
        goal['image_desired_goal'] = self.goalim
        return goal
      
    def set_goal(self, goal):
        self.goalim = goal

    def sample_goal(self):
        start_id = 9 + self.targetobj*3
        if self.door:
            start_id = 9 + self.targetobj*7
        elif self.drawer:
            start_id = 18 + self.targetobj*7
        qpos = self.data.qpos.flat.copy()
        ogpos = qpos[start_id:(start_id+2)]
        goal_pos = np.random.uniform(
                -0.2,
                0.2,
                size=(2,),
            )
        self._state_goal = goal_pos 
        self._set_obj_xyz(goal_pos) 
        self.goalim = self.render().flatten()
        self.goalst = np.concatenate([self.get_endeff_pos(), self.data.qpos[9:]])
        self._reset_hand()
        self._set_obj_xyz(ogpos)

    def sample_goals(self, bs):
        self.reset()
        ims = []
        for i in range(bs):
          self.sample_goal()
          ims.append(self.goalim)
        ims = np.stack(ims)
        return {'image_desired_goal': ims}
          
    def initialize(self):
        self.epcount = -1 # to ensure the first episode starts with 0 idx
        self.cur_path_length = 0
        self.hand_memory = []
        self.obj_memory0 = []
        self.obj_memory1 = []
        self.obj_memory2 = []
        if self.interaction:
            self.block0_interaction = []
            self.block1_interaction = []
            self.block2_interaction = []

    def reset_model(self, no_reset=False, add_noise=False):
        ''' For logging '''
        if self.verbose:
            if self.epcount % self.log_freq == 0:
                self.save_distribution()
                if self.interaction:
                    self.save_block_interaction()
                # reset the memories
                self.hand_memory = []
                self.obj_memory0 = []
                self.obj_memory1 = []
                self.obj_memory2 = []
                if self.interaction:
                    self.block0_interaction = []
                    self.block1_interaction = []
                    self.block2_interaction = []
                self.save_gif()
        self.cur_path_length = 0
        self.epcount += 1
        
        if not no_reset: # reset initial block pos
            self._reset_hand()
            for _ in range(100):
                self.do_simulation([0.0, 0.0])
            self.targetobj = np.random.randint(3)
            self.sample_goal()
            self.cur_path_length = 0

            for i in range(3):
                self.targetobj = i
                if self.randomize:
                    init_pos = np.random.uniform(
                    -0.2,
                    0.2,
                    size=(2,),
                )
                elif self.hard:
                    if i == 0:
                        init_pos = [-.2, 0] #-0.15]
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
                    if self.tower or self.double_target:
                        if i == 0:
                            init_pos = [-0.15, 0.8, 0.075]
                        if i == 1:
                            init_pos = [-0.12, 0.6, 0.075]
                        if i == 2:
                            init_pos = [0.25, 0.4, 0.075]
                    object_qvel = self.sim.data.get_joint_qvel('objGeom{}_x'.format(i))
                    object_qvel[:] = 0.
                    self.sim.data.set_joint_qvel('objGeom{}_x'.format(i), object_qvel)
                elif self.drawer:
                    if i == 0:
                        init_pos = [0.4, 0.3, 0.005]
                    if i == 1:
                        init_pos = [-0.12, 0.6, 0.005]
                    if i == 2:
                        init_pos = [0.4, 0.5, 0.005]
                    if i == 3:
                        init_pos = [-0.15, 0.4, 0.005]
                    if i == 4:
                        init_pos = [0.5, 0.4, 0.005]
                    if i == 5:
                        init_pos = [-0.2, 0.7, 0.005]
                    object_qvel = self.sim.data.get_joint_qvel('objGeom{}_x'.format(i))
                    object_qvel[:] = 0.
                    self.sim.data.set_joint_qvel('objGeom{}_x'.format(i), object_qvel)
                if add_noise:
                    init_pos += np.random.uniform(-0.02, 0.02, (2,))
                
                self.obj_init_pos = init_pos
                self._set_obj_xyz(self.obj_init_pos)
                # tower pos needs to be initialized via set_joint_qpos
                if self.door or self.drawer:
                    object_qpos = self.sim.data.get_joint_qpos('objGeom{}_x'.format(i))
                    object_qpos[:3 ] = init_pos
                    object_qpos[3:] = 0.
                    self.sim.data.set_joint_qpos('objGeom{}_x'.format(i), object_qpos)
        if self.door or self.double_target:
            self.change_door_angle(0.0)
        elif self.drawer:
            self.data.qpos[-1] = -0.05
        self.sim.forward()
        
        o = self.get_obs()
        
        if self.epcount % self.log_freq == 0:
            self.imgs = []
            im = self.sim.render(64, 64, camera_name='cam0')
            self.imgs.append(im)
            #cv2.imwrite(self.filepath + '/init.png', (cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).astype(np.uint8))
        
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
        elif self.drawer:
            low_dim_info = {'block0_x': self.data.get_site_xpos('obj0')[0], 
                                      'block0_y': self.data.get_site_xpos('obj0')[1], 
                                      'block0_z': self.data.get_site_xpos('obj0')[2], 
                                      'block1_x': self.data.get_site_xpos('obj1')[0], 
                                      'block1_y': self.data.get_site_xpos('obj1')[1], 
                                      'block1_z': self.data.get_site_xpos('obj1')[2], 
                                      'block2_x': self.data.get_site_xpos('obj2')[0], 
                                      'block2_y': self.data.get_site_xpos('obj2')[1], 
                                      'block2_z': self.data.get_site_xpos('obj2')[2], 
                                      'block3_x': self.data.get_site_xpos('obj3')[0], 
                                      'block3_y': self.data.get_site_xpos('obj3')[1], 
                                      'block3_z': self.data.get_site_xpos('obj3')[2], 
                                      'block4_x': self.data.get_site_xpos('obj4')[0], 
                                      'block4_y': self.data.get_site_xpos('obj4')[1], 
                                      'block4_z': self.data.get_site_xpos('obj4')[2], 
                                      'block5_x': self.data.get_site_xpos('obj5')[0], 
                                      'block5_y': self.data.get_site_xpos('obj5')[1], 
                                      'block5_z': self.data.get_site_xpos('obj5')[2], 
                                      'drawer': self.data.qpos[-1],
                                      'hand_x': self.get_endeff_pos()[0],
                                      'hand_y': self.get_endeff_pos()[1],
                                      'hand_z': self.get_endeff_pos()[2],
                                      'dist': - self.compute_reward()}

         #   return o, { 'green_x': self.data.get_site_xpos('obj0')[0], 
         #               'green_y': self.data.get_site_xpos('obj0')[1], 
         #               'green_z': self.data.get_site_xpos('obj0')[2], 
         #               'pink_x': self.data.get_site_xpos('obj1')[0], 
         #               'pink_y': self.data.get_site_xpos('obj1')[1], 
         #               'pink_z': self.data.get_site_xpos('obj1')[2], 
         #               'blue_x': self.data.get_site_xpos('obj2')[0], 
         #               'blue_y': self.data.get_site_xpos('obj2')[1], 
         #               'blue_z': self.data.get_site_xpos('obj2')[2], 
         #               'door_joint': self.data.qpos[-1],
         #               'hand_x': self.get_endeff_pos()[0],
         #               'hand_y': self.get_endeff_pos()[1],
         #               'hand_z': self.get_endeff_pos()[2],
         #               'dist': - self.compute_reward()}
        return o, low_dim_info 

# this wouldn't be reached anymore
        return o, {'green_x': self.data.qpos[9], 
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

    def _reset_hand(self, pos=None):
        if pos is None:
            pos = self.hand_init_pos.copy()
        for _ in range(100):
            self.data.set_mocap_pos('mocap', pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False

    # Can probably delete this
    def reset_hand(self, grip_quat, fixed=False, grip_pos=None):
        '''If need to fix grip_quat position, use this.'''
        if fixed:
            pos = grip_pos
        else:
            pos = self.hand_init_pos.copy()
        for _ in range(100):
            self.data.set_mocap_pos('mocap', pos)
            self.data.set_mocap_quat('mocap', grip_quat)
            self.do_simulation([-1,1], self.frame_skip)
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def compute_reward(self):
        if self.exploration_only: # if not goal-conditioned
            return 0.0
        if self.door:
            start_id = 9 + self.targetobj*7
        else:
            start_id = 9 + self.targetobj*3
        qpos = self.data.qpos.flat.copy()
        ogpos = qpos[start_id:(start_id+2)]
        dist = np.linalg.norm(ogpos - self._state_goal)
        return - dist
      
    def is_goal(self):
        d = self.compute_reward()
        if (d < 0.08):
            return 1
        else:
            return 0
        
    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        return statistics

    def log_diagnostics(self, paths = None, logger = None):
        pass
    
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
            block_0_pos = self.data.qpos[9:12]
            block_1_pos = self.data.qpos[16:19]
            block_2_pos = self.data.qpos[23:26]
            if self.tower or self.double_target:
                block_0_pos = [-0.15, 0.8, 0.075]
                block_1_pos = [-0.12, 0.6, 0.075]
                block_2_pos = [0.25, 0.4, 0.075]
            gripper_pos = self.sim.data.get_geom_xpos('handle')
            if self.double_target and block is not None:
                block_1_pos[:2] += np.random.uniform(-.05, 0.05, (2,))
                gripper_pos = block_1_pos.copy()
                gripper_pos[:2] += np.random.uniform(-0.02, 0.02, (2,))
                gripper_pos[-1] += np.random.uniform(-0.01, 0.01, (1,))
            goal_pos = np.concatenate([gripper_pos, block_0_pos, block_1_pos, block_2_pos])
            
        elif self.drawer:
            angle = np.random.uniform(0.05, 0.15)
            angle = -angle
            if fixed_angle is not None:
                angle = fixed_angle
            block_0_pos  = [0.4, 0.3, 0.0]
            block_1_pos = [-0.12, 0.6, 0]
            block_2_pos = [0.4, 0.5, 0]
            block_3_pos = [-0.15, 0.4, 0]
            block_4_pos = [0.5, 0.4, 0]
            block_5_pos = [-0.2, 0.7, 0]
            # block_0_pos = [-0.15, 0.8, 0.075]
            # block_1_pos = [-0.12, 0.6, 0.075]
            # block_2_pos = [0.25, 0.4, 0.075]
            # block_3_pos = [-0.15, 0.4, 0.075]
            # block_4_pos = [0.2, 0.6, 0.075]
            # block_5_pos = [-0.2, 0.7, 0.075]
            gripper_pos = self.data.get_site_xpos('handleStart')
#             gripper_pos = self.sim.data.get_geom_xpos('handle')
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
            while np.linalg.norm(block_0_pos - block_1_pos) < 0.06 or np.linalg.norm(block_0_pos - block_2_pos) < 0.06: # ensure the blocks do not overlap
                block_0_pos = np.random.uniform( # green block
                        (-.22, -0.16, 0.0),  
                        (-.18, -0.14, 0.20), 
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
                block_0_pos = [-.2, -.15, 0] + np.random.uniform(-0.02, 0.02, (3,))# pink block
                block_2_pos = [.2, -.1, 0] + np.random.uniform(-0.02, 0.02, (3,)) # blue block
                block_1_pos = np.random.uniform( 
                        (-.12, 0.13, 0.0),  
                        (-.08, 0.17, 0.20), 
                        size=(3,)) 
            while np.linalg.norm(block_1_pos - block_0_pos) < 0.06 or np.linalg.norm(block_1_pos - block_2_pos) < 0.06: # ensure the blocks do not overlap
                block_1_pos = np.random.uniform( 
                        (-.15, 0.13, 0.0),  
                        (-.05, 0.17, 0.20), 
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
                block_0_pos = [-.2, -.15, 0] + np.random.uniform(-0.02, 0.02, (3,))# pink block
                block_1_pos = [-.1, .15, 0] + np.random.uniform(-0.02, 0.02, (3,))# pink block
                block_2_pos = np.random.uniform( # green block
                        (.18, -0.12, 0.0),  
                        (.22, -0.08, 0.20), 
                        size=(3,)) 
            while np.linalg.norm(block_2_pos - block_1_pos) < 0.06 or np.linalg.norm(block_2_pos - block_0_pos) < 0.06: # ensure the blocks do not overlap
                block_2_pos = np.random.uniform( # green block
                    (.15, -0.12, 0.0),  
                    (.25, -0.08, 0.20), 
                    size=(3,)) 
            block_2_pos += np.random.uniform(-0.02, 0.02, (3,))
            gripper_pos = block_2_pos.copy()
            gripper_pos += np.random.uniform(-0.02, 0.02, (3,))
            gripper_pos[1] += 0.6
            gripper_pos[2] = np.random.uniform(0.0, 0.20)
            goal_pos = np.concatenate((gripper_pos, block_0_pos, block_1_pos, block_2_pos))
        
        if self.door or self.drawer:
            goal_img = self.save_goal_img(None, goal_pos, 0, angle=angle)
        else:
            goal_img = self.save_goal_img(None, goal_pos, 0)
        return goal_img
    
    ''' Logging Code: Saves gifs of every log_freq episode, heat maps of gripper and block positions, and plots of gripper-block distances '''
    def save_img(self):
        im = self.sim.render(64, 64, camera_name ='cam0')
        return im

    def take_steps_and_render(self, obs, actions, savename):
        '''Returns image after having taken actions from obs.'''
        threshold = 0.05
        repeat = True
        _iters = 0
        self.reset()
        while repeat:
            for i in range(6):
                if not self.drawer and i > 2:
                    break
                self.targetobj = i
                if self.door or self.drawer: 
                    self.obj_init_pos = obs[(i+1)*3:((i+1)*3)+3]
                    self._set_obj_xyz(self.obj_init_pos)
                    object_qpos = self.sim.data.get_joint_qpos('objGeom{}_x'.format(i))
                    object_qpos[:3 ] = self.obj_init_pos
                    object_qpos[3:] = 0.
                    self.sim.data.set_joint_qpos('objGeom{}_x'.format(i), object_qpos)
                    object_qvel = self.sim.data.get_joint_qvel('objGeom{}_x'.format(i))
                    object_qvel[:] = 0.
                    self.sim.data.set_joint_qvel('objGeom{}_x'.format(i), object_qvel)
                else:
                    self.obj_init_pos = obs[(i+1)*3:((i+1)*3)+2]
                    self._set_obj_xyz(self.obj_init_pos)
            if not self.door:
                error = np.linalg.norm(obs[3:12] - self.data.qpos[9:18])
                repeat = (error >= threshold)
                _iters += 1
            else:
                break
            if _iters > 10:
                break
        repeat = True
        _iters = 0
        if self.door: 
            self.change_door_angle(obs[-1])
            door_vel = np.array([0.])
            self.sim.data.set_joint_qvel('doorjoint', door_vel)
#         elif self.drawer:
#             self.data.qpos[-1] = angle
#             door_vel = np.array([0.])
#             self.sim.data.set_joint_qvel('handle', door_vel)
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
        if savename == None: 
            return im

        with imageio.get_writer(
                savename + '.gif', mode='I') as writer:
            for e in range(actions.shape[0] + 1):
                writer.append_data(imgs[e])
        return im
        
    def _restore(self):
        '''For resetting the env without having to call reset() (i.e. without updating episode count)'''
        self._reset_hand()
#         pos = obs[:3]
#         for _ in range(100): # Move gripper to pos
#             self.data.set_mocap_pos('mocap', pos)
#             self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
#             self.do_simulation([-1,1], self.frame_skip)
#         rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
#         self.init_fingerCOM  =  (rightFinger + leftFinger)/2
#         self.pickCompleted = False
        
        for i in range(3):
            self.targetobj = i
            if self.hard:
                if i == 0:
                    init_pos = [-.2, 0] #-0.15]
                elif i == 1:
                    init_pos = [-.1, .15]
                else:
                    init_pos = [ .2, -.1]
            else:
                init_pos = [0.1 * (i-1), 0.15]
            if self.door:
                init_pos = [-0.15, 0.75, 0.05 * (i+1)]
                init_pos[:2] += np.random.normal(loc=0, scale=0.001, size=2)
            
                if self.tower or self.double_target:
                    if i == 0:
                        init_pos = [-0.15, 0.8, 0.075]
                    if i == 1:
                        init_pos = [-0.12, 0.6, 0.075]
                    if i == 2:
                        init_pos = [0.25, 0.4, 0.075]
            elif self.drawer:
                if i == 0: # teal
                    init_pos = [0.4, 0.3, 0.005]
                if i == 1:
                    init_pos = [-0.12, 0.6, 0.005]
                if i == 2:
                    init_pos = [0.4, 0.5, 0.005]
                if i == 3:
                    init_pos = [-0.15, 0.4, 0.005]
                if i == 4: # olive
                    init_pos = [0.5, 0.4, 0.005]
                if i == 5:
                    init_pos = [-0.2, 0.7, 0.005]
                object_qvel = self.sim.data.get_joint_qvel('objGeom{}_x'.format(i))
                object_qvel[:] = 0.
                self.sim.data.set_joint_qvel('objGeom{}_x'.format(i), object_qvel)
            self.obj_init_pos = init_pos
            
#             object_qpos = self.sim.data.get_joint_qpos('objGeom{}_x'.format(i))
#             object_qpos[:3] = init_pos
#             object_qpos[3:] = 0.
            if self.door: 
                self.sim.data.set_joint_qpos("objGeom{}_x".format(i), object_qpos)
            self._set_obj_xyz(self.obj_init_pos)

        imgs = []
        im = self.sim.render(64, 64, camera_name='cam0')
        return {'green_x': self.data.qpos[9], 
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

    def save_goal_img(self, PATH, goal, eps, step_thru=False, actions=None, angle=None):
        '''Returns image with a given goal array of positions for the gripper and blocks.'''
        # Move end effector to green block by simulation
        if angle is not None:
            print("angle", angle)
            self.change_door_angle(angle)
        if self.drawer:
#             print(self.data.get_site_xpos('handleStart'))
            goal[:3] = self.data.get_site_xpos('handleStart')
        pos = goal[:3]
#         print("init pos", pos)
        for _ in range(100):
            self.data.set_mocap_pos('mocap', pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False

        #  Move blocks to correct positions
        for i in range(6):
            if not self.drawer and i >= 3:
                break
            self.targetobj = i
            init_pos = None
            if self.stack or self.door or self.drawer:
                init_pos = goal[(i+1)*3:((i+1)*3)+3]
                self.obj_init_pos = goal[(i+1)*3:((i+1)*3)+3]
            else:
                self.obj_init_pos = goal[(i+1)*3:((i+1)*3)+2]
            self.obj_init_pos[:2] += np.random.normal(loc=0, scale=0.001, size=2)
                
            self._set_obj_xyz(self.obj_init_pos)

            if self.door or self.drawer:
                object_qpos = self.sim.data.get_joint_qpos('objGeom{}_x'.format(i))
                object_qpos[:3] = init_pos
                object_qpos[3:] = 0.
                self.sim.data.set_joint_qpos('objGeom{}_x'.format(i), object_qpos)
                object_qvel = self.sim.data.get_joint_qvel('objGeom{}_x'.format(i))
                object_qvel[:] = 0.
                self.sim.data.set_joint_qvel('objGeom{}_x'.format(i), object_qvel)
            self.sim.forward()
        # if step_thru: 
            # only step thru actions if the flag is set to True & actions is not None
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
                
    def save_distribution(self):
        ''' Saves the heat maps for hand and block positions '''
        def draw(var_type, var1, var2, name1, name2):
            if name1 == 'X' and name2 == 'Y':
                if var_type == 0:
                    val_range = [[-0.2, 0.2],[0.4, 0.8]]
                else:
                    val_range = [[-0.3, 0.4],[-0.2, 0.3]]
            if name1 == 'X' and name2 == 'Z':
                if var_type == 0:
                    val_range = [[-0.2, 0.2], [0.0, 0.2]]
                else:
                    val_range = [[-0.3, 0.3], [0.0, 0.3]]
            if name1 == 'Y' and name2 =='Z':
                if var_type == 0:
                    val_range = [[0.4, 0.8], [0.0, 0.2]]
                else:
                    val_range = [[-0.2, 0.3], [0.0, 0.3]]
            # transform to density
            
            Xrange = np.arange(val_range[0][0]-0.05, val_range[0][1]+0.05, 0.05)
            Yrange = np.arange(val_range[1][0]-0.05, val_range[1][1]+0.05, 0.05)
            bins = Xrange, Yrange
            var1 = list(var1)
            var2 = list(var2)
            H, xedges, yedges = np.histogram2d(var1, var2, bins=bins, range=val_range, density=False)
            H = H / sum(sum(H))

            # set consistent color bar
            ax2dhist = plt.axes()
            bounds = np.linspace(0.0, 1.0, 25)
            colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(bounds)+1))
            cmap = mcolors.ListedColormap(colors[1:-1])
            cmap.set_over(colors[-1])
            cmap.set_under(colors[0])
            norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds)-1)
            X, Y = np.meshgrid(Xrange, Yrange)
            ax2dhist.pcolormesh(X, Y, np.swapaxes(H,0,1)) #, cmap=cmap)
            sm = ScalarMappable(norm=norm) #, cmap=cmap)
            sm.set_array([])
            plt.colorbar(sm)
            if var_type == 0:
                var = 'gripper'
            elif var_type == 1:
                var = 'block0'
            elif var_type == 2:
                var = 'block1'
            else:
                var = 'block2'
     
            name = name1 + name2 + var + 'pos' + str(int(self.epcount))
            plt.title(name)
            plt.savefig(self.filepath + '/' + name + '.png', bbox_inches='tight')
            plt.close()
        
        # 0: gripper, 1: block0, 2: block1, 3: block2
        for i in range(4):
            if i == 0:
                memory = np.stack(self.hand_memory)
            elif i == 1:
                memory = np.stack(self.obj_memory0)
            elif i == 2:
                memory = np.stack(self.obj_memory1)
            elif i == 3:
                memory = np.stack(self.obj_memory2)
            x = memory[:,0]
            y = memory[:,1]
            if i != 0:
                y -= 0.6
            z = memory[:,2]
            draw(i, x, y, 'X', 'Y')
            if i == 0:
                draw(i, x, z, 'X', 'Z')
                draw(i, y, z, 'Y', 'Z')
                
                
    def save_block_interaction(self):
        '''Saves the block interaction (i.e. distance between gripper and block for each block) for an episode.'''
        name = 'GripperBlockDistance' + str(int(self.epcount))
        fig = plt.figure()
        plt.plot(self.block0_interaction, "-b", label="block0", linewidth=0.5)
        plt.plot(self.block1_interaction, "-r", label="block1", linewidth=0.5)
        plt.plot(self.block2_interaction, "-m", label="block2", linewidth=0.5)
        fig.suptitle(name)
        plt.xlabel('Step')
        plt.ylabel('Distance')
        plt.legend(loc="upper right")
        plt.savefig(self.filepath + '/' + name + '.png')
        plt.close()
