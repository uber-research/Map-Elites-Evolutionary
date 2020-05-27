# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import ArgumentParser
import os
import logging

import numpy as np

from config import setup_config
from es_modular.es_manager import ESPopulationManager
from es_modular.ga_manager import GAPopulationManager

# set multi-threading of numpy to 1
# if not done, every parallel processes tries to access other processes
# this results in a massive slow down
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Sometimes mujoco locks for some reason.. this is a fix.
if os.path.exists('/home/flowers/mujoco-py/mujoco_py/generated/mujocopy-buildlock.lock'):
    os.remove('/home/flowers/mujoco-py/mujoco_py/generated/mujocopy-buildlock.lock')


"""
MAIN SCRIPT
The default parameters are set in config.py
The argument of this script override the default configuration.
"""

parser = ArgumentParser()
parser.add_argument('--trial_id', type=int, default=0, help="trial identifier (int)")
parser.add_argument('--algo', type=str, default='mees_explore_exploit', help="'mees_explore_exploit', 'mees_explore', 'mees_exploit', 'mega', 'nses', 'nsres' or 'nsraes' ")
parser.add_argument('--env_id', type=str, default='HumanoidDeceptive-v2', help="'AntMaze-v2', 'HumanoidDeceptive-v2' or 'DamageAnt-v2'")
parser.add_argument('--log_dir', type=str, default='./results/')
parser.add_argument('--config', type=str, default='default', help="also 'mees_damage', 'mees_explore' and 'custom'.")
args = parser.parse_args()

# define logger and save output
logging.basicConfig(level=logging.INFO, filename=args.log_dir + 'out.logs')
logger = logging.getLogger(__name__)
logging.getLogger().addHandler(logging.StreamHandler())


def run_main(config):
    logger.info(config)  # print config in logs
    np.random.seed(config['master_seed'])  # set master_seed

    if config['algo'] == 'mega':
        optimizer = GAPopulationManager(args=config)
    else:
        optimizer = ESPopulationManager(args=config)

    # run optimization
    optimizer.optimize(iterations=config['n_iterations'])


if __name__ == '__main__':
    # Override default configuration with arguments
    # Params are stored into a dict and saved in args.log_dir
    config = setup_config(args)
    run_main(config)
