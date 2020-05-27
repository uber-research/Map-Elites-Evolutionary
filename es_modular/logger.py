# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
from collections import namedtuple
import logging

logger = logging.getLogger(__name__)


class CSVLogger:
    def __init__(self, fnm, col_names):
        logger.info('Creating data logger at {}'.format(fnm))
        self.fnm = fnm
        self.col_names = col_names
        with open(fnm, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(col_names)
        # hold over previous values if empty
        self.vals = {name: None for name in col_names}

    def log(self, **cols):
        self.vals.update(cols)
        # logger.info(pformat(self.vals))
        if any(key not in self.col_names for key in self.vals):
            for k in self.vals:
                if k not in self.col_names:
                    print(k)
            raise Exception('CSVLogger given invalid key')
        with open(self.fnm, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([self.vals[name] for name in self.col_names])


log_fields = ['iteration',  # generations
              'pop_id',  # id of the cell in mees, of the population in nses and co
              'time_elapsed_so_far',
              'learning_progress',
              'overall_best_eval_returns_mean',
              'overall_best_eval_returns_median',
              'overall_best_eval_returns_std',
              'overall_best_eval_returns_max',
              'overall_best_eval_novelty_max',
              'overall_best_eval_novelty_mean',
              'overall_best_eval_novelty_median',
              'overall_best_eval_novelty_std',
              'overall_best_eval_len_mean',
              'overall_best_eval_len_std',
              'episodes_so_far',
              'po_returns_mean',
              'po_returns_median',
              'po_returns_std',
              'po_returns_max',
              'po_returns_min',
              'po_len_mean',
              'po_len_std',
              'po_len_max',
              'noise_std',
              'learning_rate',
              'eval_returns_mean',
              'eval_returns_median',
              'eval_returns_std',
              'eval_returns_max',
              'eval_len_mean',
              'eval_len_std',
              'eval_n_episodes',
              'eval_novelty_mean',
              'eval_novelty_std',
              'eval_novelty_median',
              'eval_novelty_max',
              'theta_norm',
              'grad_norm',
              'update_ratio',
              'episodes_this_step',
              'timesteps_this_step',
              'timesteps_so_far',
              'accept_theta_in',
              'p_exploit',  # probability to select explore in MEES (for adaptive)
              'explore',  # wheter this generation was performed with an explore or an exploir objective (for mees)
              'nsra_weight'  # adaptive weight for nsraes
              ]

StepStats = namedtuple('StepStats', ['po_returns_mean',
                                     'po_returns_median',
                                     'po_returns_std',
                                     'po_returns_max',
                                     'po_novelty_mean',
                                     'po_novelty_std',
                                     'po_novelty_median',
                                     'po_novelty_max',
                                     'po_returns_min',
                                     'po_len_mean',
                                     'po_len_std',
                                     'po_len_max',
                                     'noise_std',
                                     'learning_rate',
                                     'theta_norm',
                                     'grad_norm',
                                     'update_ratio',
                                     'episodes_this_step',
                                     'timesteps_this_step',
                                     ])

EvalStats = namedtuple('EvalStats', ['eval_returns_mean',
                                     'eval_returns_median',
                                     'eval_returns_std',
                                     'eval_returns_max',
                                     'eval_len_mean',
                                     'eval_len_std',
                                     'eval_n_episodes',
                                     'eval_novelty_mean',
                                     'eval_novelty_std',
                                     'eval_novelty_median',
                                     'eval_novelty_max'
                                     ])

POResult = namedtuple('POResult', ['noise_inds',
                                   'returns',
                                   'lengths',
                                   'bcs',
                                   'obs_sum',
                                   'obs_sq',
                                   'obs_count',
                                   'time',
                                   'final_xpos'
                                   ])
EvalResult = namedtuple('EvalResult', ['returns', 'lengths', 'bcs', 'final_xpos'])
