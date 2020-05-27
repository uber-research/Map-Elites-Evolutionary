# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
logger = logging.getLogger(__name__)
from es_modular.interaction.model import ControllerAndEnv, simulate

"""
Class that contains the environment interaction and the policy.
Environment configs can be found below.
"""

class Interaction:
    def __init__(self, args):
        self.model = ControllerAndEnv(args)
        self.init = args['policy_args']['init']
        self.max_episode_length = None

    def initial_theta(self, seed=None):
        if self.init == 'normc':
            return self.model.get_normc_model_params(seed=seed)
        elif self.init == 'zeros':
            import numpy as np
            return np.zeros(self.model.param_count)
        else:
            raise NotImplementedError('Undefined initialization scheme `{}`'.format(self.init))

    @property
    def env(self):
        return self.model.env

    def rollout(self, theta, random_state, eval=False, obs_mean=None, obs_std=None, render=False):

        # update obs stats from master
        if self.model.use_norm_obs:
            self.model.set_obs_stats(obs_mean, obs_std)
        seed = random_state.randint(1e6)

        total_return, length, bc, final_xpos, obs_sum, obs_sq, obs_count = simulate(theta,
                                                                                    self.model,
                                                                                    max_episode_length=self.max_episode_length,
                                                                                    train_mode=not eval,
                                                                                    seed=seed,
                                                                                    render=render)

        return total_return, length, bc, final_xpos, obs_sum, obs_sq, obs_count



    def rollout_batch(self, thetas, batch_size, random_state, eval=False,
                      obs_mean=None, obs_std=None, render=False):
        import numpy as np
        returns = np.zeros(batch_size)
        lengths = np.zeros(batch_size, dtype='int')
        bcs = [None] * batch_size
        final_xpos = np.zeros(batch_size)
        for i, theta in enumerate(thetas):
            returns[i], lengths[i], bcs[i], final_xpos[i], obs_sum, obs_sq, obs_count = self.rollout(theta,
                                                                                                     random_state=random_state,
                                                                                                     eval=eval,
                                                                                                     obs_mean=obs_mean,
                                                                                                     obs_std=obs_std,
                                                                                                     render=render)

        return returns, lengths, np.array(bcs), final_xpos, obs_sum, obs_sq, obs_count


class Humanoid(Interaction):
    def __init__(self, args):
        args['policy_args']['input_size'] = 376
        args['policy_args']['output_size'] = 17
        Interaction.__init__(self, args)
        self.dim_bc = 2  # dimension of the behavioral characterization
        self.min_max_bcs = [[-50, 50], [-50, 50]]  # bounds of BC space
        self.nb_cells_per_dimension = 91  # number of splits of BC space
        self.max_episode_length = 1000


class AntMaze(Interaction):
    def __init__(self, args):
        args['policy_args']['input_size'] = 29
        args['policy_args']['output_size'] = 8
        Interaction.__init__(self, args)
        self.dim_bc = 2
        self.min_max_bcs = [[-40, 40], [-40, 40]]
        self.nb_cells_per_dimension = 10
        self.max_episode_length = 3000


class Ant(Interaction):
    def __init__(self, args):
        args['policy_args']['input_size'] = 27
        args['policy_args']['output_size'] = 8
        Interaction.__init__(self, args)
        self.dim_bc = 4
        self.min_max_bcs = [[0, 1]] * 4
        self.nb_cells_per_dimension = 10
        self.max_episode_length = 1000


def build_interaction_module(env_id, args):
   if 'HumanoidDeceptive' in env_id:
        return Humanoid(args=args)
   elif 'AntMaze' in env_id:
        return AntMaze(args=args)
   elif 'DamageAnt' in env_id:
       return Ant(args=args)
   else:
       raise NotImplementedError
