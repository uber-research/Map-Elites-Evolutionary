# Modifications Copyright (c) 2020 Uber Technologies Inc.
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os

"""
Modified from https://github.com/openai/mlsh from 
the "Meta-Learning Shared Hierarchies" paper: https://arxiv.org/abs/1710.09767
"""

class AntObstaclesBigEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.goal = np.array([35, -25])
        this_path = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, this_path + '/assets/ant_obstaclesbig2.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        reward = - np.sqrt(np.sum(np.square(self.data.qpos[:2] - self.goal)))
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(bc=self.data.qpos[:2],
                                      x_pos=self.data.qpos[0])

    def _get_obs(self):
        qpos = self.data.qpos.flatten()
        qpos[:2] = (qpos[:2] - 5) / 70
        return np.concatenate([
            qpos,
            self.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.8
        self.viewer.cam.elevation = -60
        self.viewer.cam.lookat[0] = 14
        self.viewer.cam.lookat[1] = -4
        self.viewer.opengl_context.set_buffer_size(1280, 1024)
