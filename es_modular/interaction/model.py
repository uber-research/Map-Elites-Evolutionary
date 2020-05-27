# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import time
import logging
logger = logging.getLogger(__name__)

import numpy as np
import gym
from es_modular.interaction import custom_gym

"""
Class containing the policy. 
Interaction happen in the simulate function below.
The model might use virtual batch normalization (normalization of the observations). In that case it tracks the observation statistics and normalize
before they are used by the policy.
"""

class ObsStats:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.std = np.ones(shape, dtype=np.float32)

    def set_from_init(self, init_mean, init_std):
        if init_mean is not None:
            self.mean = init_mean
            self.std = init_std


class RunningStat:
    def __init__(self, shape):
        self.sum = np.zeros(shape, dtype=np.float32)
        self.sumsq = np.zeros(shape, dtype=np.float32)
        self.count = 0

    def increment(self, s, ssq, c):
        self.sum += s
        self.sumsq += ssq
        self.count += c

class ControllerAndEnv:
    ''' simple feedforward model '''

    def __init__(self, args):
        self.env_id = args['env_id']
        self.layer_1 = args['policy_args']['layers'][0]
        self.layer_2 = args['policy_args']['layers'][1]
        self.input_size = args['policy_args']['input_size']
        self.output_size = args['policy_args']['output_size']
        self.shapes = [(self.input_size, self.layer_1),
                       (self.layer_1, self.layer_2),
                       (self.layer_2, self.output_size)]
        self.action_noise = float(args['policy_args']['action_noise'])
        self.use_norm_obs = args['env_args']['use_norm_obs']

        if  args['policy_args']['activation'] == 'tanh':
            self.activations = [np.tanh, np.tanh, np.tanh]
        else:
            raise NotImplementedError('Unknown activation function')

        self.weight = []
        self.bias = []
        self.param_count = 0
        for shape in self.shapes:
            self.weight.append(np.zeros(shape=shape))
            self.bias.append(np.zeros(shape=shape[1]))
            self.param_count += (np.product(shape) + shape[1])

        self.render_mode = False
        # if not without_env:
        self.make_env()
        if self.use_norm_obs:
            self.obs_stats = ObsStats(self.env.observation_space.shape)

    def make_env(self, render_mode=False):
        self.render_mode = render_mode
        self.env = gym.make(self.env_id)

    def get_action(self, x):
        h = np.array(x).flatten()
        num_layers = len(self.weight)
        for i in range(num_layers):
            w = self.weight[i]
            b = self.bias[i]
            h = np.tanh(np.matmul(h, w) + b)
        return h

    def set_model_params(self, model_params):
        pointer = 0
        for i in range(len(self.shapes)):
            w_shape = self.shapes[i]
            b_shape = self.shapes[i][1]
            s_w = np.product(w_shape)
            s = s_w + b_shape
            chunk = np.array(model_params[pointer:pointer + s])
            self.weight[i] = chunk[:s_w].reshape(w_shape)
            self.bias[i] = chunk[s_w:].reshape(b_shape)
            pointer += s

    def set_obs_stats(self, obs_mean, obs_std):
        self.obs_stats.set_from_init(obs_mean, obs_std)

    def get_normc_model_params(self, seed, stdev=0.01):
        if seed is not None:
            np.random.seed(seed)
        # sample weigths from normal distribution and set biases to 0
        random_params = np.zeros([self.param_count])
        pointer = 0
        n_biases = 0
        for i in range(len(self.shapes)):
            # sample weights
            w_shape = self.shapes[i]
            s_w = np.product(w_shape)
            assert len(w_shape) == 2
            out = np.random.randn(w_shape[0], w_shape[1]).astype(np.float32)
            out *= stdev / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            random_params[pointer:pointer + s_w] = out.flatten()
            pointer += s_w
            # setting biases to 0
            b_shape = self.shapes[i][1]
            random_params[pointer:pointer + b_shape] = 0
            pointer += b_shape
            n_biases += b_shape
        assert pointer == self.param_count, "Something is wrong with the normal initialization"
        assert np.argwhere(random_params == 0).size == n_biases, "Something is wrong with the normal initialization"
        return np.asarray(random_params, dtype=np.float32)


def simulate(theta, model, max_episode_length, seed, train_mode=False,
             render=False):

    # set the parameters to run in the controller class
    model.set_model_params(theta)

    # seed the environment (new seed each time)
    if seed >= 0:
        # logger.debug('Setting seed to {}'.format(seed))
        random.seed(seed)
        np.random.seed(seed)
        model.env.seed(seed)

    if model.use_norm_obs:
        running_stats = RunningStat(model.env.observation_space.shape)

    obs = model.env.reset()

    # track obs stats, but not every rollout for computation efficiency
    record_obs = False
    if model.use_norm_obs and train_mode:
        if np.random.rand() < 0.01:
            record_obs = True
            all_obs = [obs]

    total_reward = 0.0
    final_pos = None
    t = 0
    info = {}
    for t in range(max_episode_length):
        if model.use_norm_obs:
            # use obs_stats that are updated by master
            obs = np.asarray((obs - model.obs_stats.mean) / model.obs_stats.std, dtype=np.float32)

        # get action from model
        action = model.get_action(obs)

        # add action noise
        if train_mode and model.action_noise != 0:
            action += np.random.randn(action.size) * model.action_noise

        # make step in env
        obs, reward, done, info = model.env.step(action)

        # track final x position
        final_pos = info['x_pos']

        # performance might be computed as the last reward or the sum of rewards depending on the environments
        if model.env_id == 'AntMaze-v2':
            total_reward = reward
        else:
            total_reward += reward

        if render:
            model.env.render()

        if record_obs:
            all_obs.append(obs)

        if done:
            break
    bc = info['bc']

    if record_obs:
        # update running stats
        all_obs = np.array(all_obs)
        sum_obs = all_obs.sum(axis=0)
        count = all_obs.shape[0]
        sum_obs_sq = np.square(all_obs).sum(axis=0)
        running_stats.increment(sum_obs, sum_obs_sq, count)

    if model.use_norm_obs:
        sum = running_stats.sum
        sumsq = running_stats.sumsq
        count = running_stats.count
    else:
        sum = sumsq = count = None

    return total_reward, t, bc, final_pos, sum, sumsq, count

