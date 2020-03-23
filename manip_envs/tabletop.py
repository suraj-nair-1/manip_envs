from collections import OrderedDict
import numpy as np
from gym.spaces import  Dict , Box
import math
import os
import torch

from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv

from pyquaternion import Quaternion
from metaworld.envs.mujoco.utils.rotation import euler2quat
import cv2

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
            hand_init_pos = (0, 0.6, 0.2),
            liftThresh = 0.04,
            rewMode = 'orig',
            rotMode='rotz',
            problem="rand",
            exploration = "hard",
            low_dim=False,
            **kwargs
    ):
        self.randomize = False
        self.exploration = exploration
        self.quick_init(locals())
        hand_low=(-0.2, 0.4, 0.0)
        hand_high=(0.2, 0.8, 0.20)
        obj_low=(-0.3, 0.4, 0.1)
        obj_high=(0.3, 0.8, 0.3)
        SawyerXYZEnv.__init__(
            self,
            frame_skip=5,
            action_scale=1./10,
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

        self.epcount = 0
        self.epsucc = []
        self.lowdim = low_dim
        self.liftThresh = liftThresh
        self.max_path_length = 100
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
            self.observation_space = Box(0, 1.0, (3+9,))
        else:
            self.observation_space = Box(0, 1.0, (64,64,3))
            
        self.goal_space = self.observation_space

    @property
    def model_name(self):
      dirname = os.path.dirname(__file__)
      if self.exploration == "easy":
        filename = os.path.join(dirname, "../assets/sawyer_xyz/sawyer_multiobject.xml")
      else:
        filename = os.path.join(dirname, "../assets/sawyer_xyz/sawyer_multiobject_hard.xml")
      return filename

    def step(self, action):
        self.set_xyz_action_rotz(action[:4])
        self.do_simulation([action[-1], -action[-1]])

        ob = None
        ob = self.get_obs()
        reward  = self.compute_reward()
        self.curr_path_length +=1
        if self.curr_path_length == self.max_path_length:
            self._reset_hand()
            done = True
        else:
            done = False
        return ob, reward, done, {'pos': ob, 'hand': self.get_endeff_pos(), 'success':self.is_goal()}
   
    def get_obs(self, goal=False):
        
        if self.lowdim:
            if goal:
              return self.goalst
            obj = self.data.qpos[9:]
            gpos = self.get_endeff_pos()
            obs = np.concatenate([gpos, obj])
        else:
            if goal:
                return self.goalim
            im = self.sim.render(64, 64, camera_name="cam0")
            obs = im
        return obs

    def _get_info(self):
        pass
    
    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        start_id = 9 + self.targetobj*3
        qpos[start_id:(start_id+2)] = pos.copy()
        qvel[start_id:(start_id+2)] = 0
        self.set_state(qpos, qvel)

    def render(self, mode=""):
        return self.sim.render(64, 64, camera_name="cam0")

    def sample_goal(self):
        start_id = 9 + self.targetobj*3
        qpos = self.data.qpos.flat.copy()
        ogpos = qpos[start_id:(start_id+2)]
        goal_pos = np.random.uniform(
                -0.3,
                0.3,
                size=(2,),
            )
        self._state_goal = goal_pos 
        self._set_obj_xyz(goal_pos) 
        self.goalim = self.sim.render(64, 64, camera_name="cam0")
        self.goalst = np.concatenate([self.get_endeff_pos(), self.data.qpos[9:]])
        self._reset_hand()
        self._set_obj_xyz(ogpos)

    def reset_model(self):
        self._reset_hand()

        buffer_dis = 0.04
        block_pos = None
        for i in range(3):
            self.targetobj = i
            if self.randomize:
              init_pos = np.random.uniform(
                  -0.2,
                  0.2,
                  size=(2,),
              )
            else:
              init_pos = [0.1 * (i-1), 0.0]
            self.obj_init_pos = init_pos
            self._set_obj_xyz(self.obj_init_pos)

        for _ in range(100):
          self.do_simulation([0.0, 0.0])
        self.targetobj = np.random.randint(3)
        self.sample_goal()
        self.curr_path_length = 0
        self.epcount += 1
        o = self.get_obs()
        
        #Can try changing this
        return o

    def _reset_hand(self):
        pos = self.hand_init_pos.copy()
        for _ in range(10):
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
