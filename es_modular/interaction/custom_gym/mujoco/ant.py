# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from es_modular.interaction.custom_gym.mujoco.damage_for_adaptability_tests import modify_ant_xml
from mujoco_py.generated import const
import os

class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, modif='', param=None):
        self.step_count = 0
        self.bc_scores = np.zeros(4)
        this_path = os.path.dirname(os.path.abspath(__file__))
        if modif != '':
            modify_ant_xml(this_path + '/assets/', modif=modif, param=param)
            mujoco_env.MujocoEnv.__init__(self, this_path + '/assets/ant_modified.xml', 5)
        else:
            mujoco_env.MujocoEnv.__init__(self, this_path + '/assets/ant.xml', 5)
            utils.EzPickle.__init__(self)


    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        bc = self.update_bc()

        return ob, reward, done, dict(reward_forward=forward_reward,
                                      reward_ctrl=-ctrl_cost,
                                      reward_contact=-contact_cost,
                                      reward_survive=survive_reward,
                                      x_pos=xposafter,
                                      bc=bc)


    def update_bc(self):
        # bc is the proportion of time each of the leg touches the floor
        self.step_count += 1
        contacts = self.sim.data.contact
        ncon = self.sim.data.ncon
        for i, c in enumerate(contacts[:ncon]):
            geom1 = self.sim.model.geom_id2name(c.geom1)
            geom2 = self.sim.model.geom_id2name(c.geom2)
            if geom1 == 'floor':
                if 'left_ankle_geom' in geom2:
                    self.bc_scores[0] += 1
                elif 'right_ankle_geom' in geom2:
                    self.bc_scores[1] += 1
                elif 'third_ankle_geom' in geom2:
                    self.bc_scores[2] += 1
                elif 'fourth_ankle_geom' in geom2:
                    self.bc_scores[3] += 1
        bc = self.bc_scores / self.step_count
        return bc

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.bc_scores = np.zeros(4)
        self.step_count = 0

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 2
        self.viewer.cam.fixedcamid += 1
        self.viewer.cam.type = const.CAMERA_FIXED
        self.viewer.opengl_context.set_buffer_size(1280, 1024)