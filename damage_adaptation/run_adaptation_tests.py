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

import json
import numpy as np

from es_modular.interaction.interaction import *
from es_modular.interaction.custom_gym.mujoco.test_adapted_envs import damaged_ant_env_ids
from config import CONFIG_DEFAULT
from damage_adaptation.mboa import run_M_BOA_procedure

logger = logging.getLogger(__name__)

"""
Run adaptation experiments from a behavioral map built via Map-Elites.
"""

# Result folder from which to run the damage recovery experiment
FOLDER = '/path_to_results/'

# Parameters
DATA_ALREADY_EXTRACTED = True
RHO = 0.03  # the higher the value, the greater the portion of the performance map will be affected by each evaluation
SIGMA2_NOISE = 0.01  # prior on the noise in performance evaluation
KAPPA = 0.3  # exploration factor to scale the 'uncertainty bonus' in the choice of the next controller to evaluate.
N_ITERATIONS = 30
N_EVALS = 2  # number of evaluation rollouts (to compute performance estimate).

def run_damage_recovery_experiments(results_path=FOLDER):
    policy_folder = results_path + '/policies/'
    os.makedirs(results_path + 'me_archive_adaptation', exist_ok=True)
    data_folder = results_path + 'me_archive_adaptation/'

    list_envs = damaged_ant_env_ids

    # extract the controllers and their stats into simple text files.
    if not DATA_ALREADY_EXTRACTED:
        thetas = []
        bcs = []
        perfs = []
        obs_mean = []
        obs_std = []
        policy_files = os.listdir(policy_folder)
        cell_ids = []
        for i, policy_file in enumerate(policy_files):
            cell_id = int(policy_file[:-5])
            cell_ids.append(cell_id)
    
            with open(policy_folder + policy_file, 'rt') as f:
                out = json.load(f)
    
            thetas.append(out['theta'])
            bcs.append(out['bc'])
            perfs.append(out['performance'])
            obs_mean.append(out['obs_mean'])
            obs_std.append(out['obs_std'])
            print(i / len(policy_files), ' %')
    
        thetas = np.arra(thetas)
        bcs = np.array(bcs)
        perfs = np.array(perfs)
        obs_mean = np.array(obs_mean)
        obs_std = np.array(obs_std)
        cell_ids = np.array(cell_ids)
        np.savetxt(data_folder + '/bcs.txt', bcs)
        np.savetxt(data_folder + '/perfs.txt', perfs)
        np.savetxt(data_folder + '/thetas.txt', thetas)
        np.savetxt(data_folder + '/obs_mean.txt', obs_mean)
        np.savetxt(data_folder + '/obs_std.txt', obs_std)
        np.savetxt(data_folder + '/cell_ids.txt', cell_ids)
    else:
        # load data
        bcs = np.loadtxt(data_folder + '/bcs.txt')
        perfs = np.loadtxt(data_folder + '/perfs.txt')
        thetas = np.loadtxt(data_folder + '/thetas.txt')
        obs_mean = np.loadtxt(data_folder + '/obs_mean.txt')
        obs_std = np.loadtxt(data_folder + '/obs_std.txt')
        cell_ids = np.loadtxt(data_folder + '/cell_ids.txt')
    
    
    candidate_cell_ids = []
    candidate_perfs_after_damage = []
    candidate_perfs_before_damage = []
    best_perf_after_damage_all = []
    
    
    n_cells = thetas.shape[0]
    best_cell_id = np.argmax(perfs)
    best_perf = np.max(perfs)
    print('Best score on undamaged agent: {}, from cell id {}'.format(best_perf, best_cell_id))
    
    
    # loop over all damages defined in interaction.custom_gym.mujoco.test_adapted_envs
    config = CONFIG_DEFAULT
    for env_id in list_envs:
        print('\n\n\n\t\t\t', env_id, '\n\n')
    
        # build env with damage
        config.update(env_id=env_id)
        env = build_interaction_module(env_id=env_id, args=config)
        rs = np.random.RandomState()
    
        # run the Map-Based Bayesian Optimization Algorithm
        out = run_M_BOA_procedure(env, rs, bcs, perfs, thetas, obs_mean, obs_std, cell_ids,
                                  n_iterations=N_ITERATIONS,
                                  rho=RHO,  # parameter for the kernel function computing the covariance matrix of the GP (better = BC affect each other further away)
                                  kappa=KAPPA,  # exploration parameter for the GP, it tries the policy such as argmax(predicted_perf + KAPPA * uncertainty)
                                  sigma2_noise=SIGMA2_NOISE,  # predicted noise on performance
                                  n_evals=N_EVALS,
                                  best_cell_id=best_cell_id,  # id of the best policy before damage, as reference
                                  )
        best_perf_damaged, candidate_cell_id, candidate_perfs = out
        candidate_cell_ids.append(cell_ids[candidate_cell_id])
        candidate_perfs_after_damage.append(candidate_perfs)
        candidate_perfs_before_damage.append(perfs[candidate_cell_id])
        best_perf_after_damage_all.append(best_perf_damaged)
    
        np.savetxt(data_folder + 'candidate_ids.txt', candidate_cell_ids)  # cell_id of the recovery policy for each of the damages
        np.savetxt(data_folder + 'candidate_perfs_afterdamage.txt', candidate_perfs_after_damage) # performances of the recovery policies after damage (10 perf / damage)
        np.savetxt(data_folder + 'candidate_perfs_beforedamage.txt', candidate_perfs_before_damage) # performances of the recovery policies before damage (10 perfs / damage)
        np.savetxt(data_folder + 'formerbest_id_and_perfs_beforedamage.txt', [int(cell_ids[best_cell_id]), best_perf])  # performance of the former best policy before damage
        np.savetxt(data_folder + 'formerbest_perfs_afterdamage.txt', best_perf_after_damage_all)  # performance of the former best policy after damage.
    
        print('Performance of best policy', int(cell_ids[best_cell_id]), ' from the archive on undamaged robot: ', best_perf)
        print('Performance of best policy from the archive on damaged robot: ', np.mean(best_perf_damaged))
        print('Performance of candidate policy', int(cell_ids[candidate_cell_id]), ' from the archive on undamaged robot: ', perfs[candidate_cell_id])
        print('Performance of candidate policy from the archive on damaged robot: ', np.mean(candidate_perfs))

parser = ArgumentParser()
parser.add_argument('--results_dir', type=str, default='', help="trial identifier (int)")
if __name__ == '__main__':
    args = parser.parse_args()
    run_damage_recovery_experiments(args.results_dir)