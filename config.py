# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import os

import json
import numpy as np

from es_modular.utils import find_save_path
"""
CONFIG_DEFAULT is a default configuration that can be run locally (although it does not solve any problem as such).

To reproduce results from the 'Scaling Map-Elites to Deep Neuroevolution' paper, the CONFIG_MEES_PAPER should be used.
It uses 1000 workers, which requires the use of a cluster, and the Fiber dependency.
Link to the paper: https://arxiv.org/pdf/2003.01825.pdf
Link to the fiber repo: https://github.com/uber/fiber/

Note that the number of rollout per generation is num_workers * batch_size * 2 (* 2  because of mirror sampling)
"""


CONFIG_DEFAULT = dict(n_iterations=1000,  # number of generations
                      master_seed=np.random.randint(int(1e6)),
                      num_workers=2,  # number of workers
                      batch_size=1,  # number of perturbations per worker per generation (each perturb leads to 2 rollouts because of mirror sampling).
                      eval_batches_per_step=2,  # number of rollouts for evaluation
                      eval_batch_size=1,  # number of evaluation rollout per worker. Leave to 1, as you have many more workers than evluation rollouts anyway
                      noise_std=0.02,  # standard deviation of the Gaussian noise applied on the parameters for ES updates
                      fitness_normalization='centered_ranks',
                      env_args=dict(use_norm_obs=True),  # whether to use virtual batch normalization (computing running stats for observations and normalizing the obs)
                      optimizer_args=dict(optimizer='adam',  # either adam or sgd
                                          learning_rate=0.01,
                                          l2_coeff=0.005,
                                          divide_gradient_by_noise_std=False),  # OpenAI ES paper and POET use this, not sure why
                      policy_args=dict(init='normc',
                                       layers=[256, 256],
                                       activation='tanh',
                                       action_noise=0.01),
                      novelty_args=dict(k=10,  # number of nearest neighbors for novelty search objective
                                        nses_selection_method='nov_prob',  # 'nov' prob selects the pop proportionally to its novelty score, also 'robin' (one after the other)
                                        nses_pop_size=5),  # number of populations
                      mees_args=dict(nb_consecutive_steps=10,  # number of consecutive ES steps before choosing a new starting cell for MEES
                                     explore_strategy='novelty_bias',  # method to bias starting cell sampling for exploration steps. Also 'best_or_uniform'
                                     exploit_strategy='best2inlast5_or_best',  # method to bias starting cell sampling for exploration steps. Also 'best_or_uniform'
                                     strategy_explore_exploit='robin')  # strategy for balancing between exploration and exploitation.
                      )

# Configuration for the damage recovery experiment
CONFIG_MEES_PAPER_DAMAGE = CONFIG_DEFAULT.copy()
CONFIG_MEES_PAPER_DAMAGE.update(env_id='DamageAnt-v2',
                                n_iterations=500,
                                num_workers=1000,
                                batch_size=5,
                                eval_batches_per_step=30,
                                eval_batch_size=1)
CONFIG_MEES_PAPER_DAMAGE['mees_args'].update(explore_strategy='best_or_uniform',
                                             exploit_strategy='best_or_uniform')

# Configuration for the deep exploration experiment
CONFIG_MEES_PAPER_EXPLORATION = CONFIG_DEFAULT.copy()
CONFIG_MEES_PAPER_EXPLORATION.update(env_id='DeceptiveHumanoid-v2',
                                     n_iterations=1000,
                                     num_workers=1000,
                                     batch_size=5,
                                     eval_batches_per_step=30,
                                     eval_batch_size=1)
CONFIG_MEES_PAPER_EXPLORATION['mees_args'].update(explore_strategy='novelty_bias',
                                                  exploit_strategy='best2inlast5_or_best')

# Custom configuration
CONFIG_CUSTOM = CONFIG_DEFAULT.copy()


def setup_config(args):
    args_dict = vars(args)

    if args_dict['config'] == 'default':
        config_dict = CONFIG_DEFAULT
        config_dict.update(args_dict)
    elif args_dict['config'] == 'https://arxiv.org/pdf/2003.01825.pdf':
        config_dict = CONFIG_MEES_PAPER_DAMAGE
        env_error_msg = "The damage recovery experiment of the MEES paper was conducted with the DamageAnt-v2 domain"
        assert args_dict['env_id'] == 'DamageAnt-v2', env_error_msg
    elif args_dict['config'] == 'mees_exploration':
        config_dict = CONFIG_MEES_PAPER_EXPLORATION
        env_error_msg = "The deep exploration experiment of the MEES paper was conducted with either 'DeceptiveHumanoid-v2' or 'AntMaze-v2'"
        assert args_dict['env_id'] in ['HumanoidDeceptive-v2', 'AntMaze-v2'], env_error_msg
        config_dict['env_id'] = args_dict['env_id']
    elif args_dict['config'] == 'custom':
        config_dict = CONFIG_CUSTOM
    else:
        raise NotImplementedError
    config_dict['algo'] = args_dict['algo']
    config_dict['time'] = str(datetime.datetime.now())  # save date and time of job.
    config_dict['log_dir'] = find_save_path(args_dict['log_dir'], args_dict['trial_id'])  # add +100 to trial_id if it already exists
    with open(config_dict['log_dir'] + 'config.json', 'wt') as f:
        json.dump(config_dict, f)

    return config_dict
