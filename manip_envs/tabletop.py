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
            hand_init_pos = (0, 0.6, 0.2),
            liftThresh = 0.04,
            rewMode = 'orig',
            rotMode='rotz',
            problem="rand",
            door = True, #Add door to the env
            exploration = "hard",
            low_dim=False, #True,
            filepath="test",
            max_path_length=50,
            verbose=1,
            hard=False,
            log_freq=100, # in terms of episode num
            smm=True, #False,
            exploration_only=False,
            **kwargs
    ):
        self.randomize = False
        self.smm = smm
        self.door = door # if True, add door to the env
        self._hard = hard # if True, blocks are initialized to diff corners
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

        self.imsize= 48
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
                self.imsize = 48 #64
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
            filename = os.path.join(dirname, "../assets/sawyer_xyz/sawyer_multiobject.xml")
        else:
            if self.door:
                filename = os.path.join(dirname, "../assets/sawyer_xyz/sawyer_multiobject_door_v1.xml")
            else:
                filename = os.path.join(dirname, "../assets/sawyer_xyz/sawyer_multiobject_hard.xml")
        return filename

    def change_door_angle(self, angle):
        old_jt = self.data.qpos.copy()[-1]
        self.data.qpos[-1] = angle
        print("Door joint before: {} | now: {}".format(old_jt, self.data.qpos[-1]))
    
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
            block0 = self.data.get_geom_xpos('objGeom0')[:3].copy()
            block1 = self.data.get_geom_xpos('objGeom1')[:3].copy()
            block2 = self.data.get_geom_xpos('objGeom2')[:3].copy()
            self.hand_memory.append(hand)
            self.obj_memory0.append(block0)
            self.obj_memory1.append(block1)
            self.obj_memory2.append(block2)
                
            if self.epcount % self.log_freq == 0:
                im = self.sim.render(48, 48, camera_name='cam0')
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
        # print('target ob id')
        # print(self.targetobj)
        # print('qpos')
        # print(qpos)
        start_id = 9 + self.targetobj*3
        if len(pos) < 3:
            qpos[start_id:(start_id+2)] = pos.copy()
            qvel[start_id:(start_id+2)] = 0
        else:
            qpos[start_id:(start_id+3)] = pos.copy()
            qvel[start_id:(start_id+3)] = 0
        # print('qpos of interest')
        # print(qpos[start_id:(start_id+2)])
        self.set_state(qpos, qvel)

    def render(self, mode=""):
        i =  self.sim.render(self.imsize, self.imsize, camera_name="cam0")  / 255. #cam0
        i = np.swapaxes(i, 0, 2)
        return i
      
    def get_goal(self):
        goal  = {}
        goal['image_desired_goal'] = self.goalim
        return goal
      
    def set_goal(self, goal):
        self.goalim = goal

    def sample_goal(self):
        start_id = 9 + self.targetobj*3
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

    def reset_model(self):
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
        self._reset_hand()
        self.epcount += 1
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
            elif self._hard:
                if i == 0:
                    init_pos = [-.25, -0.2]
                elif i == 1:
                    init_pos = [-.25, .2]
                else:
                    init_pos = [ .15, -.2]
            else:
                init_pos = [0.1 * (i-1), 0.15]
            self.obj_init_pos = init_pos
            self._set_obj_xyz(self.obj_init_pos)

        
        for _ in range(100):
            self.do_simulation([0.0, 0.0])
        self.targetobj = np.random.randint(3)
        self.sample_goal()
        self.cur_path_length = 0
        o = self.get_obs()
        
        if self.epcount % self.log_freq == 0:
            self.imgs = []
            im = self.sim.render(48, 48, camera_name='cam0')
            self.imgs.append(im)
            #cv2.imwrite(self.filepath + '/init.png', (cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).astype(np.uint8))

        #Can try changing this
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

    def _reset_hand(self):
        pos = self.hand_init_pos.copy()
        for _ in range(10):
            self.data.set_mocap_pos('mocap', pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False

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
    
    
    ''' Logging Code: Saves gifs of every log_freq episode, heat maps of gripper and block positions, and plots
        of gripper-block distances. 
    '''
    def save_img(self, PATH, eps, step):
        im = self.sim.render(48, 48, camera_name ='cam0')
        return im

    
    def take_steps_and_render(self, obs, actions, savename):
        '''Returns image after having taken actions from obs.'''
        # print("inside take steps and render")
        threshold = 0.05
        repeat = True
        _iters = 0
        while repeat:
            for i in range(3):
                self.targetobj = i
                self.obj_init_pos = obs[(i+1)*3:((i+1)*3)+2]
                self._set_obj_xyz(self.obj_init_pos)
            error = np.linalg.norm(obs[3:12] - self.data.qpos[9:18])
            repeat = (error >= threshold)
            _iters += 1
            if _iters > 10:
                break
        repeat = True
        _iters = 0
        while repeat:
            pos = obs[:3]
            for _ in range(100): # Move gripper to pos
                self.data.set_mocap_pos('mocap', pos)
                self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
                self.do_simulation([-1,1], self.frame_skip)
            rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
            self.init_fingerCOM  =  (rightFinger + leftFinger)/2
            self.pickCompleted = False
            error = np.linalg.norm(pos - self.get_endeff_pos())
            repeat = (error >= threshold)
            _iters += 1
            if _iters > 10:
                break
        imgs = []
        im = self.sim.render(48, 48, camera_name='cam0')
        imgs.append(im)
        for i in range(actions.shape[0]):
            action = actions[i]
            self.set_xyz_action_rotz(action[:4])
            self.do_simulation([action[-1], -action[-1]])
            im = self.sim.render(48, 48, camera_name='cam0')
            imgs.append(im)
            
        im = self.sim.render(48, 48, camera_name='cam0')
        
        with imageio.get_writer(
                savename + '.gif', mode='I') as writer:
            for e in range(actions.shape[0] + 1):
                writer.append_data(imgs[e])
        return im
        
    def _restore(self, obs):
        for i in range(3):
            self.targetobj = i
            self.obj_init_pos = obs[(i+1)*3:((i+1)*3)+2]
            self._set_obj_xyz(self.obj_init_pos)
        pos = obs[:3]
        for _ in range(100): # Move gripper to pos
            self.data.set_mocap_pos('mocap', pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False
        print(self.data.qpos[9:18])
        imgs = []
        im = self.sim.render(48, 48, camera_name='cam0')

    def save_goal_img(self, PATH, goal, eps):
        '''Returns image with a given goal array of positions for the gripper and blocks.'''
        for i in range(3):
            self.targetobj = i
            self.obj_init_pos = goal[(i+1)*3:((i+1)*3)+3]
            self._set_obj_xyz(self.obj_init_pos)
            
        # Move end effector to green block by simulation
        pos = goal[:3]
        for _ in range(100):
            self.data.set_mocap_pos('mocap', pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False
        
        im = self.sim.render(48, 48, camera_name='cam0')
        return im

    
    def save_gif(self):
        ''' Saves the gif of an episode.
        '''
        
        with imageio.get_writer(
                self.filepath + '/Eps' + str(self.epcount) + '.gif', mode='I') as writer:
            for i in range(self.max_path_length + 1):
                writer.append_data(self.imgs[i])

        #with imageio.get_writer(
        #        self.filepath + '/Eps' + str(self.epcount) + '.gif', mode='I') as writer:
        #    writer.append_data(imageio.imread(self.filepath + '/init.png'))
        #    for i in range(self.max_path_length):
        #        img_path = self.filepath + '/obs' + str(i) + '.png'
        #        writer.append_data(imageio.imread(img_path))
                
    def save_distribution(self):
        ''' Saves the heat maps for hand and block positions.
        '''
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
