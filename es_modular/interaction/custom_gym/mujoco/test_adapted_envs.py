# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import gym
from es_modular.interaction import custom_gym
import numpy as np


def run_env(id):
    print(id)
    env = gym.make(id)
    n_act = env.action_space.shape[0]
    env.reset()
    for _ in range(300):
        act = np.random.uniform(-1, 1, n_act)
        env.step(act)

damaged_ant_env_ids = []
for joints in [[0], [1], [2], [3], [4], [5], [6], [7], [0, 1], [2, 3], [4, 5], [6, 7],
    [2, 3, 4, 5], [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 4, 5], [0, 1, 6, 7], [2, 3, 6, 7],
               [0, 2], [4, 6], [0, 2, 4, 6], [1, 3], [5, 7]]:
    for i in range(len(joints)):
        joints[i] = str(joints[i])
    string = ''.join(joints)
    damaged_ant_env_ids.append('DamageAntBrokenJoint{}-v2'.format(string))
