# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import pickle
import os

import json
import numpy as np
from sklearn.neighbors import NearestNeighbors

from es_modular.stats import compute_centered_ranks
from es_modular.utils import fix_probas
logger = logging.getLogger(__name__)


class BCArchive:
    """
    The archive is actually two archives:
     - The archive of all past BC. It is used to compute novelty measures
     - The current map-elite population (Behavioral map).
    """
    def __init__(self, args):
        self.args = args
        self.k = args['novelty_args']['k']  # number of neighbors for knn novelty score
        self.dim_bc = args['env_args']['dim_bc']  # number of dimensions of the bc
        self.nb_cells_per_dimension = args['env_args']['nb_cells_per_dimension']  # number of cells per dimension for map discretization
        self.min_max_bcs = args['env_args']['min_max_bcs']  # min and max values for each dimension of the bc
        self.use_norm_obs = args['env_args']['use_norm_obs']  # whether observation normalization is ued or not (bool)
        self.algo = args['algo']
        
        self.archive_folder = args['log_dir'] + '/archive/'
        os.makedirs(self.archive_folder, exist_ok=True)
        self.policy_folder = args['log_dir'] + '/policies/'
        os.makedirs(self.policy_folder, exist_ok=True)

        self.nn_model = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree', metric='euclidean')
        self.all_bcs = []  # contains bcs of all thetas seen so far
        self.all_perfs = []  # contains performances of thetas seen so far
        self.empty = True  # wheter the archive is empty or not

        self.nses_strategy = args['novelty_args']['nses_selection_method']  # strategy to select between the different population of nses-like algorithms
        self.nses_pop_size = args['novelty_args']['nses_pop_size']  # number of populations for nses like algorithms
        self.current_pop_id = 0  # used for round robin selection of nses-like algorithm population
        self.nses_population_bcs = np.zeros([self.nses_pop_size, self.dim_bc])  # contains the current bcs for the current populations of nses-like algos
        self.nses_population_novelties = np.ones([self.nses_pop_size]) * 1e3  # contains novelty score for each population of nses-like algos, optimistic init
        self.nses_population_performances = np.zeros([self.nses_pop_size])  # contains performance of each population of nses-like algos

        # # # # # # #
        # ME archive
        # # # # # # # 
        
        self.mees_strategy_explore = args['mees_args']['explore_strategy']  # strategy to sample a cell to start exploration
        self.mees_strategy_exploit = args['mees_args']['exploit_strategy']  # strategy to sample a cell to start exploitation
        self.cell_ids = np.arange(self.nb_cells_per_dimension ** self.dim_bc).reshape([self.nb_cells_per_dimension] * self.dim_bc)  # array containing all cell ids
        self.filled_cells = []  # list of filled cells, contain cell ids
        self.iteration_filled_cells = []
        self.me_bcs = []  # list of stored policies bcs
        self.performances = []  # list of stored policies performances
        # keep tracks of various metrics
        self.stats_novelty = [[]]
        self.stats_xpos = [[]]
        self.stats_best_train = [[]]
        self.stats_archive_performance = dict(mean=[], std=[], median=[], max=[], min=[], cell_count=[])

        # Define boundaries
        self.boundaries = []  # boundaries for each cell
        self.cell_sizes = []  # compute cell size
        for i in range(self.dim_bc):
            bc_min = self.min_max_bcs[i][0]
            bc_max = self.min_max_bcs[i][1]
            boundaries = np.arange(bc_min, bc_max + 1e-5, (bc_max - bc_min) / self.nb_cells_per_dimension)
            boundaries[0] = - np.inf
            boundaries[-1] = np.inf
            self.boundaries.append(boundaries)
            self.cell_sizes.append((bc_max - bc_min) / self.nb_cells_per_dimension)

        # log
        self.history = []
        os.makedirs(self.archive_folder, exist_ok=True)
        with open(self.archive_folder + 'cell_ids.pk', 'wb') as f:
            pickle.dump(self.cell_ids, f)
        with open(self.archive_folder + 'cell_boundaries.pk', 'wb') as f:
            pickle.dump(self.boundaries, f)

    def compute_novelty(self, bc):
        if bc.ndim == 1:
            bc = bc.reshape(1, -1)
        if self.empty:
            distances = (np.ones([bc.shape[0]]) * np.inf).reshape(-1, 1)
        else:
            distances, _ = self.nn_model.kneighbors(bc, n_neighbors=min(self.k, self.size_ns))
            # skip the bc if it is already in the archive.
            if distances.shape[0] == 1 and distances[0, 0] == 0:
                distances = distances[:, 1:]
        return (distances ** 2).mean(axis=1)

    @property
    def size_ns(self):
        return len(self.all_bcs)

    @property
    def size_me(self):
        return len(self.filled_cells)

    def sample(self, explore):
        # function called to sample the next population to optimize, the method depends on the algorithm
        if not self.empty:
            if self.algo in ['nses', 'nsres', 'nsraes']:
                pop_id = self.sample_nses_nsres_nsraes()
                theta = None
                bc = self.nses_population_bcs[pop_id, :]
                perf = self.nses_population_performances[pop_id]
            elif 'me' in self.algo:
                pop_id, theta, bc, perf = self.sample_mees(explore)
            else:
                raise NotImplementedError
            return pop_id, theta, bc, perf
        else:
            raise IndexError('The archive is empty')

    def sample_nses_nsres_nsraes(self):
        if self.nses_strategy == 'random':
            pop_id = np.random.randint(self.nses_pop_size)
        elif self.nses_strategy == 'robin':
            pop_id = (self.current_pop_id + 1) % self.nses_pop_size
            self.current_pop_id = pop_id
        elif self.nses_strategy == 'nov_prob':
            novelty_probs = self.nses_population_novelties / self.nses_population_novelties.sum()
            pop_id = np.random.choice(range(self.nses_pop_size), 1, p=novelty_probs)[0]
            logger.info('Pop ' + str(pop_id) + ' sampled. Novelty probabilities:' + str(novelty_probs))
        else:
            raise NotImplementedError
        return pop_id

    def compute_probabilities(self, n_candidates, values, method='uniform'):
        # method to compute probabilities depending on values
        if method == 'uniform':
            probas = 1 / n_candidates * np.ones([n_candidates])
        else:
            scores = values.copy()
            n_candidates = scores.size
            eps = 0.1
            if method == 'proportional':
                probas = scores / scores.sum()
            elif method == 'uniform_plus_proportional_5_best':
                except_best_5 = np.argsort(scores)[:-5]
                scores[except_best_5] = 0
                probas = eps * np.ones(n_candidates) / n_candidates + (1 - eps) * scores / scores.sum()
            elif method == 'argmax':
                probas = np.zeros([n_candidates])
                probas[np.argmax(values)] = 1
            else:
                raise NotImplementedError
        return probas

    def sample_mees(self, explore):
        # compute probability depending on the explore/exploit state and sampling strategy
        n_candidates = len(self.filled_cells)
        p_random = 1 / n_candidates
        if (explore and self.mees_strategy_explore == 'uniform') or (not explore and self.mees_strategy_exploit == 'uniform'):
            values = None
            method = 'uniform'
        elif explore and self.mees_strategy_explore == 'novelty_bias':
            values = np.array(self.stats_novelty[-1])
            method = 'uniform_plus_proportional_5_best'
        elif (not explore and self.mees_strategy_exploit == 'best_or_uniform') or (explore and self.mees_strategy_explore == 'best_or_uniform'):
            if np.random.rand() < 0.5:
                values = np.array(self.performances)
                method = 'argmax'
            else:
                values = None
                method = 'uniform'
            p_random = 1 / n_candidates
        elif not explore and self.mees_strategy_exploit == 'best2inlast5_or_best':
            values = np.zeros([n_candidates])
            perfs = np.array(self.stats_best_train[-1]).copy()
            perfs += (1 - perfs.min())
            if np.random.rand() < 0.5 or n_candidates < 5:
                # pick the best
                sorted_inds = np.flip(np.argsort(perfs).flatten(), axis=0)
                values[sorted_inds[0]] = perfs[sorted_inds[0]]
            else:
                # pick proportionaly among 2 best of 5 last discovered cells
                last_inds = np.arange(max(perfs.size // 2, perfs.size - 5), perfs.size)
                sorted_inds = np.flip(np.argsort(perfs[last_inds]).flatten(), axis=0)
                values[last_inds[sorted_inds[:2]]] = perfs[last_inds[sorted_inds[:2]]]
            p_random = 1 / n_candidates
            method = 'proportional'
        else:
            raise NotImplementedError
        probs = self.compute_probabilities(n_candidates=n_candidates, values=values, method=method)

        # sample the next cell_id to start from
        # extremely ugly fix to avoid approx errors to ruin the probs.sum() == 1
        probs, test_passed = fix_probas(probs)

        if test_passed:
            ind = np.random.choice(range(n_candidates), p=probs)
        else:
            logger.info('Probs have fucked up' + str(probs))
            ind = np.random.choice(range(n_candidates))

        p = probs[ind]
        cell_id = self.filled_cells[ind]
        to_log = 'Pop {} sampled with probability {}. Method: {}. p_random = {}'.format(cell_id, p, method, p_random)
        logger.info(to_log)

        # extract corresponding theta, perf, bc, load from file
        perf = self.performances[ind]
        bc = self.me_bcs[ind]
        policy_file = self.policy_folder + str(cell_id) + '.json'
        with open(policy_file, 'rt') as f:
            out = json.load(f)
        theta = np.array(out['theta'])
        assert np.all(bc == np.array(out['bc']))
        assert perf == out['performance']
        return cell_id, theta, bc, perf

    def find_cell_id(self, bc):
        """
        Find cell identifier of the BC map corresponding to bc
        """
        coords = []
        for j in range(self.dim_bc):
            inds = np.atleast_1d(np.argwhere(self.boundaries[j] < bc[j]).squeeze())
            coords.append(inds[-1])
        coords = tuple(coords)
        cell_id = self.cell_ids[coords]
        return cell_id
    
    def add(self, new_bc, performance, novelty, theta, final_xpos,
            iter, explore, obs_stats, starting_bc, pop_id, previous_bc, best_train_performance):
        """
        Attempt to add the new individual to the behavioral map
        """
        if self.empty:
            self.all_bcs.append(new_bc)
            self.all_perfs.append(performance)
            self.nn_model.fit(np.array(self.all_bcs))

        # update nses and variants populations
        if self.algo in ['nses', 'nsres', 'nsraes']:
            self.nses_population_bcs[pop_id, :] = new_bc
            self.nses_population_novelties[pop_id] = novelty
            self.nses_population_performances[pop_id] = performance

        # update selection scores
        # copy old selection scores for that new generation (need to copy, as cell may be updated)
        if iter > 0 and iter == len(self.stats_best_train):
            self.stats_best_train.append(self.stats_best_train[-1].copy())
            self.stats_novelty.append(self.stats_novelty[-1].copy())
            self.stats_xpos.append(self.stats_xpos[-1].copy())

        # # # # # # # # # #
        # Update ME Archive
        # # # # # # # # # #

        # find cell_id of the new_bc
        cell_id = self.find_cell_id(new_bc)

        if iter != -1:
            # if starting from a cell, update the best training performance for that cell, This is used to bias sampling of cell to start exploitation
            previous_cell_id = self.find_cell_id(previous_bc)
            previous_ind = np.argwhere(self.filled_cells == previous_cell_id)[0][0]
            if np.all(previous_bc == self.me_bcs[previous_ind]):
                self.stats_best_train[-1][previous_ind] = best_train_performance
                # logger.info('Updating best performance of cell {}, with best perf {}'.format(previous_cell_id, best_train_performance))

        str_history = ''

        # if cell already filled
        if cell_id in self.filled_cells:
            new_cell = False
            ind = np.argwhere(self.filled_cells == cell_id)[0][0]
            # check whether performance is better
            if self.performances[ind] < performance:
                update = True
                theta_update = theta
                self.performances[ind] = performance
                self.me_bcs[ind] = new_bc
                perf_update = performance
                bc_update = new_bc
                str_history += 'perf'
            else:
                update = False
                theta_update = None
                bc_update = None
                perf_update = None

        # if new cell
        else:
            new_cell = True
            update = True
            ind = -1
            self.filled_cells.append(cell_id)
            self.iteration_filled_cells.append(iter)
            self.stats_best_train[-1].append(0)
            self.stats_xpos[-1].append(0)
            self.stats_novelty[-1].append(0)
            self.performances.append(performance)
            self.me_bcs.append(new_bc)
            theta_update = theta
            bc_update = new_bc
            perf_update = performance

        if update:
            self.save_policy(policy_dir=self.policy_folder,
                             theta=theta_update,
                             bc=bc_update,
                             performance=perf_update,
                             cell_id=cell_id,
                             obs_stats=obs_stats,
                             iter=iter
                             )

            # log
            if new_cell:
                self.history.append(['add', str_history, bc_update, perf_update, iter, cell_id, starting_bc, explore])
                to_log = 'New theta added to the archive in cell {}, new bc {}! '.format(cell_id, bc_update)
            else:
                self.history.append(['replace', str_history, bc_update, perf_update, iter, cell_id, starting_bc, explore])
                to_log = 'New theta added to the archive in cell {}. '.format(cell_id)
                if str_history == 'perf':
                    to_log += 'Better performance.'
            logger.info(to_log)

            # store best training performance of the previous cell in the newly discovered cell.
            # it will be updated when it is selected as starting cell
            self.stats_best_train[-1][ind] = best_train_performance
            self.stats_xpos[-1][ind] = final_xpos

        else:
            self.history.append(['nothing', str_history, bc_update, perf_update, iter, cell_id, starting_bc, explore])

        # update novelty scores
        if iter == -1:
            self.stats_novelty[-1][ind] = 1e6
        else:
            self.update_novelty()

        # update novelty archive and knn model
        # add bc and perf to list of all bcs and perfs
        if not self.empty:
            self.all_bcs.append(new_bc)
            self.all_perfs.append(performance)
            self.nn_model.fit(np.array(self.all_bcs))
        else:
            self.empty = False

        # track stats about the quality of the archive
        self.stats_archive_performance['mean'].append(np.mean(self.all_perfs))
        self.stats_archive_performance['median'].append(np.median(self.all_perfs))
        self.stats_archive_performance['max'].append(np.max(self.all_perfs))
        self.stats_archive_performance['std'].append(np.std(self.all_perfs))
        self.stats_archive_performance['min'].append(np.min(self.all_perfs))
        self.stats_archive_performance['cell_count'].append(len(self.all_perfs))
        return update

    def update_novelty(self):
        for ind in range(len(self.filled_cells)):
            bc = self.me_bcs[ind].reshape(1, -1)
            novelty = self.compute_novelty(bc)[0]
            self.stats_novelty[-1][ind] = novelty

    def save_data(self):

        with open(self.archive_folder + 'history.pk', 'wb') as f:
            pickle.dump(self.history, f)

        with open(self.archive_folder + 'best_train.pk', 'wb') as f:
            pickle.dump(self.stats_best_train, f)

        with open(self.archive_folder + 'final_xpos.pk', 'wb') as f:
            pickle.dump(self.stats_xpos, f)

        with open(self.archive_folder + 'nov.pk', 'wb') as f:
            pickle.dump(self.stats_novelty, f)

        with open(self.archive_folder + 'stat_archive_perfs.pk', 'wb') as f:
            pickle.dump(self.stats_archive_performance, f)

        np.savetxt(self.archive_folder + 'final_me_perfs.txt', np.array(self.performances))
        np.savetxt(self.archive_folder + 'final_ns_bcs.txt', np.array(self.all_bcs))
        np.savetxt(self.archive_folder + 'final_ns_perfs.txt', np.array(self.all_perfs))
        np.savetxt(self.archive_folder + 'final_me_bcs.txt', np.array(self.me_bcs))
        np.savetxt(self.archive_folder + 'final_filled_cells.txt', np.array(self.filled_cells))

    def save_policy(self, policy_dir, theta, bc, cell_id, performance, obs_stats, iter):
        policy_file = policy_dir + str(cell_id) + '.json'
        to_save = dict()
        if os.path.exists(policy_file):
            with open(policy_file, 'rt') as f:
                to_save = json.load(f)
        to_save.update(theta=theta.tolist(),
                       bc=bc.tolist(),
                       performance=performance,
                       iter=iter,
                       obs_mean=None,
                       obs_std=None)
        if self.use_norm_obs:
            to_save.update(obs_mean=obs_stats.mean.tolist(),
                           obs_std=obs_stats.std.tolist())

        with open(policy_file, 'wt') as f:
            json.dump(to_save, f, sort_keys=True)
