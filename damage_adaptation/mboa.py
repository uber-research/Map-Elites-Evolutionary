# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from config import *

logger = logging.getLogger(__name__)
green = [41 / 255, 125 / 255,  35 / 255]
blue = [49 / 255, 82 / 255, 138 / 255]

"""
Implements the Map-Based Bayesian Optimization Algorithm from Cully's paper 'Robots that Adapt like Animals'.
Adapted from Fabien Benureau's implementation: https://github.com/benureau/recode/blob/master/cully2015/cully2015.py

It's a Gaussian Process estimating the performance after damage given the behavioral characterization.
It uses as a prior the performance of the policies stored in the archive before damage. Then it selects which policy to play on the damaged robot 
using a criteria argmax(predicted_perf + kappa * uncertainty), then plays the corresponding policy and updates its model.
It is called by the run_adaptation_tests.py script.
"""


# we use an RBF kernels
def dist(x, y):
    return np.sqrt(sum((x_i - y_i) ** 2 for x_i, y_i in zip(x, y)))


def matern(x, y, rho):
    """Return the Matern kernel function (with nu = 5/2)"""
    d = dist(x, y)
    return (1 + 5 ** 0.5 * d / rho + 5 * d * d / (3 * rho * rho)) * np.exp(-5 ** 0.5 * d / rho)


def select_test(kappa, P_f):
    """Select the controller to try as the argmax of (mu + kappa*sigma2)"""
    max_p, cell_id_t = float('-inf'), None
    for cell_id, (m_x, sigma2_x) in P_f.items():
        p = m_x + kappa * sigma2_x
        if p > max_p:
            max_p, cell_id_t = p, cell_id

    return cell_id_t, max_p


def select_candidate(P_f):
    """Select the controller to try as the argmax of (mu)"""
    max_p, cell_id_t = float('-inf'), None
    for cell_id, (m_x, sigma2_x) in P_f.items():
        p = m_x
        if p > max_p:
            max_p, cell_id_t = p, cell_id

    return cell_id_t


def update_gaussian_process(perf_simu, bcs, tried_perfs, tried_cell_ids, tried_bcs, sigma2_noise, rho, P_f ):
    """Update the distribution of the performance"""
    P_diff = np.array([perf_i - perf_simu[cell_id_i] for perf_i, cell_id_i in zip(tried_perfs, tried_cell_ids)])

    K = np.array([[matern(x, y, rho) for x in tried_bcs] for y in tried_bcs]) + sigma2_noise * np.eye(len(tried_bcs))
    K_inv = np.linalg.pinv(K)  ## Compute the observations' correlation matrix.

    for cell_id in P_f.keys():
        behavior = bcs[cell_id, :]
        k = np.array([matern(behavior, xi_i, rho) for xi_i in tried_bcs])  ## Compute the behavior vs. observation.
        ## correlation  vector.
        mu = perf_simu[cell_id] + np.dot(k.T, np.dot(K_inv, P_diff))  ## Update the mean.
        sigma2 = matern(behavior, behavior, rho) - np.dot(np.dot(k.T, K_inv), k)  ## Update the variance.
        P_f[cell_id] = (mu, sigma2)  ## Update the Gaussian Process.

    return P_f

def evaluation_step(env, rs, thetas, obs_mean, obs_std, kappa, n_evals, P_f):
    cell_id_t, max_p = select_test(kappa, P_f)  ## Select next test (argmax of acquisition function).
    perfs_t = []
    for _ in range(n_evals):
        perfs_t.append(env.rollout(thetas[cell_id_t, :],
                                   random_state=rs,
                                   eval=True,
                                   obs_mean=obs_mean[cell_id_t, :],
                                   obs_std=obs_std[cell_id_t, :],
                                   render=False)[0])  ## Evaluation of ctrl_t on the broken robot.

    return perfs_t, cell_id_t, max_p


def normalize_perfs(perfs, min_perf, max_perf):
    perfs = perfs.copy() - min_perf
    perfs = perfs / (max_perf - min_perf)
    return perfs.copy()

def unnormalize_perfs(normalized_perfs, min_perf, max_perf):
    perfs = normalized_perfs.copy() * (max_perf - min_perf) + min_perf
    return perfs

def run_M_BOA_procedure(env,
                        rs,
                        bcs,
                        perfs,
                        thetas,
                        obs_mean,
                        obs_std,
                        cell_ids,
                        n_iterations,
                        rho,
                        kappa,
                        sigma2_noise,
                        n_evals,
                        best_cell_id):

    min_perf = perfs.min()
    max_perf = perfs.max()

    # we normalize performance before modeling them with the GP.
    normalized_perfs = normalize_perfs(perfs, min_perf, max_perf)

    n_cells = cell_ids.size
    P_f = {}  # performance probability distribution
    perf_simu = {}  # performance of the undamaged agent on the performance function

    # Initializing GP model with performances of the undamaged agent
    for cell_id in range(n_cells):
        bc = bcs[cell_id, :]
        mu = normalized_perfs[cell_id]
        sigma2   = matern(bc, bc, rho)
        P_f[cell_id] = (mu, sigma2)
        perf_simu[cell_id] = mu

    tried_cell_ids  = [] # the cells of the map whose controller has been executed on the broken robot.
    tried_bcs = [] # and corresponding the behaviors
    tried_perfs = [] # and corresponding the performances

    while len(tried_bcs) < n_iterations:  ## Iteration loop.
        perfs_t, cell_id, max_p = evaluation_step(env, rs, thetas, obs_mean, obs_std, kappa, n_evals, P_f)

        tried_cell_ids.append(cell_id)
        tried_bcs.append(bcs[cell_id, :])
        tried_perfs.append(np.mean(normalize_perfs(perfs_t, min_perf, max_perf)))

        P_f = update_gaussian_process(perf_simu, bcs, tried_perfs, tried_cell_ids, tried_bcs, sigma2_noise, rho, P_f.copy())  ## Update the Gaussian Process.

        string = '\n\nTesting policy {}: \n   From bc {} \n   Former perf {} \n   GP selection score {} \n   Current GP estimation {} \n   New perf on damaged Ant: {}'
        print(string.format(int(cell_ids[cell_id]),
                            bcs[cell_id],
                            perfs[cell_id],
                            unnormalize_perfs(max_p, min_perf, max_perf),
                            (unnormalize_perfs(P_f[cell_id][0], min_perf, max_perf),
                             P_f[cell_id][1]),
                            unnormalize_perfs(tried_perfs[-1], min_perf, max_perf)))


    # when the iteration are over, the candidate is the one whose estimated performance is maximal
    candidate_cell_id = select_candidate(P_f)

    # test the candidate over 20 rollouts
    candidate_perfs = []
    for _ in range(20):
        candidate_perfs.append(env.rollout(thetas[candidate_cell_id, :],
                                 random_state=rs,
                                 eval=True,
                                 obs_mean=obs_mean[candidate_cell_id, :],
                                 obs_std=obs_std[candidate_cell_id, :],
                                 render=False)[0])

    # test former best
    best_perf_damaged = []
    for _ in range(20):
        best_perf_damaged.append(env.rollout(thetas[best_cell_id, :],
                                 random_state=rs,
                                 eval=True,
                                 obs_mean=obs_mean[best_cell_id, :],
                                 obs_std=obs_std[best_cell_id, :],
                                 render=False)[0])

    return best_perf_damaged, candidate_cell_id, candidate_perfs
