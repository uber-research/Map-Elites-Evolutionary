# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import json
import numpy as np

from es_modular.es import ESIndividual, initialize_worker_fiber
from es_modular.bc_archive import BCArchive
from es_modular.logger import CSVLogger, log_fields
from es_modular.stats import RunningStat
from es_modular.interaction.interaction import build_interaction_module
import time

logger = logging.getLogger(__name__)

try:
    import fiber as mp
    logger.info('Using Fiber')
except:
    import multiprocessing as mp
    logger.info('Using Multiprocessing')


class ESPopulationManager:
    """
    This class organizes and trains collection of controllers, either organized in a behavioral map (for ME-ES and variants)
    or as a fixed-sized collection of controllers (NS-ES and variants).
    This collection is implemented by a dict and is called a population.
    Its keys are ids (behavioral cell ids for ME-Es and variants), and its values are called individuals.
    Each individual is associated with an ES optimizer, a parent controller theta (set of parameters) as well as
    its average BC and average performance as computed by the evaluation of theta.
    """
    def __init__(self, args):
        self.args = args
        self.interaction = build_interaction_module(env_id=args['env_id'], args=args)

        # Create fiber/multiprocessing worker pool
        mp_ctx = mp.get_context('spawn')
        manager = mp_ctx.Manager()
        self.manager = manager
        self.fiber_shared = {"theta": manager.dict(),
                             "args": args,
                             "env_id": args['env_id'],
                             "obs_mean": manager.dict(),
                             "obs_std": manager.dict()
                             }
        self.fiber_pool = mp_ctx.Pool(args['num_workers'],
                                      initializer=initialize_worker_fiber,
                                      initargs=(self.fiber_shared["theta"],
                                                self.fiber_shared["args"],
                                                self.fiber_shared['env_id'],
                                                self.fiber_shared["obs_mean"],
                                                self.fiber_shared["obs_std"]))
        logger.info('Worker pool created')

        self.algo = args['algo']
        self.use_norm_obs = args['env_args']['use_norm_obs']
        self.env_id = args['env_id']
        self.learning_rate = self.args['optimizer_args']['learning_rate']
        self.noise_std = self.args['noise_std']
        self.observation_shape = self.interaction.env.observation_space.shape

        self.population = dict()
        self.best = dict(performance=-np.inf)
        self.best_eval_stats = None

        # save information about BC space
        args['env_args'].update(nb_cells_per_dimension=self.interaction.nb_cells_per_dimension,
                                min_max_bcs=self.interaction.min_max_bcs,
                                dim_bc=self.interaction.dim_bc)
        self.bc_archive = BCArchive(args)  # create BC archive

        theta, obs_mean, obs_std = None, None, None

        # create initial populations (one if mees, several if nses or nsres)
        if self.algo in ['nses', 'nsres', 'nsraes']:
            for i in range(args['novelty_args']['nses_pop_size']):
                self.add_individual(theta=theta, pop_id=i)
        else:
            self.add_individual(theta=theta, pop_id=-1)

        # use observation normalization
        if self.use_norm_obs:
            logger.info('Using observation normalization, using running statistics.')
            self.obs_stats = RunningStat(self.observation_shape, obs_mean=obs_mean, obs_std=obs_std, eps=1e-2)
            self.old_obs_mean = self.obs_stats.mean
            self.old_obs_std = self.obs_stats.std
            self.old_obs_count = self.obs_stats.count
        else:
            self.obs_stats = None

        # logging
        log_path = args['log_dir'] + '/' + args['log_dir'].split('/')[-1] + 'results.csv'
        self.data_logger = CSVLogger(log_path, log_fields)
        self.filename_best = args['log_dir'] + '/' + args['log_dir'].split('/')[-1] + 'best_policy.json'
        self.t_start = time.time()
        self.episodes_so_far = 0
        self.timesteps_so_far = 0

        # set starting state of ME-ES algorithms
        if 'mees' in self.algo:
            # set the self.explore state variable for mees algorithm
            if self.algo == 'mees_explore':
                self.explore = True
                self.proba_exploit = 0
            elif self.algo == 'mees_explore_exploit':
                self.proba_exploit = 0.5
                self.explore = True
            elif self.algo == 'mees_exploit':
                self.explore = False
                self.proba_exploit = 1
        else:
            self.explore = None
            self.proba_exploit = None

        # set starting state of NSRA-ES
        if self.algo == 'nsraes':
            self.nsra_weight = 0.5  # initial exploitation weight
            self.nb_iter_since_last_best = 0  # counter to track number of generation without improvement
            self.nsra_delta_weight = 0.05  # step to update exploitation weight
            self.nsra_last_best_perf = -1e6  # tracking of last best performance
            self.nsra_increase_ratio = 0.1  # need to improve performance by this factor to increase exploitation weight
            self.nsra_n_before_decrease = 20  # number of generation without improvement to trigger a decrease in exploitation weight
        else:
            self.nsra_weight = None
            self.nb_iter_since_last_best = None
            self.nsra_delta_weight = None
            self.nsra_last_best_perf = None
            self.nsra_increase_ratio = None
            self.nsra_n_before_decrease = None

    def add_individual(self, theta=None, perf=None, bc=None, pop_id=None):
        # populations is a dict:
        # For nses, nsres, nsraes there are n_pop populations pop_id [0, n_pop - 1]
        # For mees, pop_id is the id of the me_cell

        if theta is None:
            theta = self.interaction.initial_theta()

        self.population[pop_id] = ESIndividual(pop_id=pop_id,
                                               algo=self.algo,
                                               args=self.args,
                                               fiber_pool=self.fiber_pool,
                                               fiber_shared=self.fiber_shared,
                                               theta=theta,
                                               bc=bc,
                                               perf=perf,
                                               bc_archive=self.bc_archive,
                                               )

    def next_generation(self, iteration, pop_id, starting_bc, previous_bc):

        t_i_step = time.time()

        individual = self.population[pop_id]
        assert individual.pop_id == pop_id

        # # # # # #
        # Training
        # # # # # #

        # synchronize obs stats
        if self.use_norm_obs:
            individual.update_obs_stats(self.obs_stats)

        # sample pseudo-offsprings and evaluate them
        training_task = individual.start_step()

        # compute update for new parent from offsprings
        updated_theta, optimizer, obs_updates, training_stats = individual.get_step(training_task=training_task,
                                                                                      explore=self.explore,
                                                                                      nsra_weight=self.nsra_weight)

        # # # # # # #
        # Evaluation
        # # # # # # #

        # evaluate the new parent (and other theta updates)
        self_eval_tasks = individual.start_theta_eval(updated_theta)
        eval_bcs, eval_returns, eval_novelties, eval_final_xpos, eval_stats = individual.get_theta_eval(new_theta=updated_theta,
                                                                                                        eval_task=self_eval_tasks)
        logger.info('Theta after update: ' + str(individual.theta[:3]))

        self.update_best(pop_id=pop_id, iteration=iteration, eval_stats=eval_stats)

        # # # # # # # # #
        # Update Archive
        # # # # # # # # #

        # update archive
        new_bc = eval_bcs.mean(axis=0)
        new_perf = eval_returns.mean()
        new_novelty = eval_novelties.mean()
        self.bc_archive.add(new_bc=new_bc.copy(),
                            performance=new_perf,
                            best_train_performance=training_stats.po_returns_max,
                            final_xpos=eval_final_xpos.mean(),
                            previous_bc=previous_bc.copy(),
                            novelty=new_novelty,
                            theta=updated_theta,
                            iter=iteration,
                            explore=self.explore,
                            obs_stats=self.obs_stats,
                            starting_bc=starting_bc,
                            pop_id=pop_id)

        # # # # # # # # # #
        # Update Obs Stats
        # # # # # # # # # #

        if self.use_norm_obs:
            for i in range(obs_updates['obs_sums'].shape[0]):
                self.obs_stats.increment(obs_updates['obs_sums'][i, :], obs_updates['obs_sqs'][i, :], obs_updates['obs_counts'][i])

        # # # # # # #
        # Update Logs
        # # # # # # #
        self.update_logs_dict(iteration=iteration,
                              training_stats=training_stats,
                              eval_stats=eval_stats,
                              t_i_step=t_i_step,
                              pop_id=pop_id
                              )

        return new_bc, new_perf

    def optimize(self, iterations):

        # evaluate initial individuals
        for pop_id in self.population.keys():
            individual = self.population[pop_id]

            # synchronize obs stats across individuals
            if self.use_norm_obs:
                individual.update_obs_stats(self.obs_stats)

            self_eval_task = individual.start_theta_eval(individual.theta)
            eval_bcs, eval_returns, eval_novelties, eval_final_xpos, eval_stats = individual.get_theta_eval(new_theta=individual.theta,
                                                                                                            eval_task=self_eval_task,
                                                                                                            start=True)

            # update archive
            self.bc_archive.add(new_bc=eval_bcs.mean(axis=0),
                                performance=eval_returns.mean(),
                                best_train_performance=eval_returns.mean(),
                                final_xpos=eval_final_xpos.mean(),
                                previous_bc=None,
                                novelty=eval_novelties.mean(),
                                theta=individual.theta,
                                iter=-1,
                                explore=self.explore,
                                obs_stats=self.obs_stats,
                                starting_bc=None,
                                pop_id=pop_id,
                                )
        to_log = 'Iter -1, initialization of NS archive to size {}, ME archive to size {}'
        logger.info(to_log.format(self.bc_archive.size_ns, self.bc_archive.size_me))

        # if mees algo, follow same theta for several steps
        # otherwise sample a new each time
        if 'mees' in self.algo:
            nb_consecutive_steps = self.args['mees_args']['nb_consecutive_steps']
        else:
            nb_consecutive_steps = 1

        iteration = 0
        while iteration < iterations:

            logger.info('\t\tSAMPLING NEW STARTING POLICY\n')
            if self.algo == 'mees_explore_exploit':
                # decide whether to explore or exploit
                self.mees_explore_or_exploit()

            # decide which theta to evolve next
            starting_pop_id, starting_theta, starting_bc, starting_perf = self.bc_archive.sample(self.explore)

            previous_bc = starting_bc.copy()
            previous_perf = starting_perf

            # in mees algo, a new individual is created each time (reset optimizer)
            if 'mees' in self.algo:
                if starting_pop_id in self.population.keys():
                    del self.population[starting_pop_id]
                self.add_individual(theta=starting_theta, bc=previous_bc.copy(), perf=previous_perf, pop_id=starting_pop_id)
                assert starting_theta[0] == self.population[starting_pop_id].theta[0]
                assert np.all(starting_bc == self.population[starting_pop_id].bc)
                assert starting_perf == self.population[starting_pop_id].perf
            else:
                starting_theta = self.population[starting_pop_id].theta
                assert np.all(starting_bc == self.population[starting_pop_id].bc)
                assert starting_perf == self.population[starting_pop_id].perf

            if self.algo == 'nsraes':
                # update nsraes weight
                if self.nb_iter_since_last_best >= self.nsra_n_before_decrease:
                    self.nb_iter_since_last_best = 0
                    self.nsra_weight = max(0, self.nsra_weight - self.nsra_delta_weight)
                    logger.info('NSRA-ES exploitation weight has been decreased: w = {}'.format(self.nsra_weight))

            for _ in range(nb_consecutive_steps):
                iteration += 1
                self.log_iteration(iteration, starting_pop_id, previous_bc, previous_perf)
                previous_bc, previous_perf = self.next_generation(iteration=iteration,
                                                                  pop_id=starting_pop_id,
                                                                  starting_bc=starting_bc.copy(),
                                                                  previous_bc=previous_bc.copy())

                # logs
                if iteration % 10 == 0:
                    # save bcs and policy
                    self.bc_archive.save_data()
                    if iteration % 100 == 0:
                        self.save_best_policy(self.filename_best + '.arxiv.' + str(iteration), self.best.copy())
                    self.save_best_policy(self.filename_best, self.best.copy())

                    # logs about obs_stats
                    if self.use_norm_obs:
                        self.log_obs_stats()

                logger.info('\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n ')

    def mees_explore_or_exploit(self):
        if self.args['mees_args']['strategy_explore_exploit'] == 'robin':
            self.explore = not self.explore
        elif self.args['mees_args']['strategy_explore_exploit'] == 'probabilistic':
            self.explore = False if np.random.rand() < 0.5 else True
        else:
            raise NotImplementedError

    def update_best(self, pop_id, iteration, eval_stats):
        """
        Update statistics of best individual
        """
        individual = self.population[pop_id]
        if individual.perf > self.best['performance']:
            self.best.update(theta=individual.theta.tolist(),
                             performance=individual.perf,
                             bc=individual.bc.tolist(),
                             iter=iteration)
            if self.use_norm_obs:
                self.best.update(obs_mean=self.obs_stats.mean.tolist(),
                                 obs_std=self.obs_stats.std.tolist())
            self.best_eval_stats = eval_stats
            logger.info('New best score: {}, BC {}, from population {}'.format(individual.perf,
                                                                               individual.bc,
                                                                               pop_id))

        if self.algo == 'nsraes':

            if individual.perf > self.nsra_last_best_perf + (np.abs(self.nsra_last_best_perf) * self.nsra_increase_ratio):
                self.nsra_last_best_perf = individual.perf
                self.nb_iter_since_last_best = 0
                self.nsra_weight = min(1, self.nsra_weight + self.nsra_delta_weight)
                logger.info('NSRA-ES exploitation weight has been increased: w = {}'.format(self.nsra_weight))
            else:
                self.nb_iter_since_last_best += 1

    def update_logs_dict(self, iteration, pop_id, training_stats, eval_stats, t_i_step):

        self.episodes_so_far += training_stats.episodes_this_step
        self.timesteps_so_far += training_stats.timesteps_this_step

        log_data = {
            'iteration': iteration,
            'po_returns_mean': training_stats.po_returns_mean,
            'po_returns_median': training_stats.po_returns_median,
            'po_returns_std': training_stats.po_returns_std,
            'po_returns_max': training_stats.po_returns_max,
            'po_returns_min': training_stats.po_returns_min,
            'po_len_mean': training_stats.po_len_mean,
            'po_len_std': training_stats.po_len_std,
            'po_len_max': training_stats.po_len_max,
            'noise_std': training_stats.noise_std,
            'learning_rate': training_stats.learning_rate,
            'eval_returns_mean': eval_stats.eval_returns_mean,
            'eval_returns_median': eval_stats.eval_returns_median,
            'eval_returns_std': eval_stats.eval_returns_std,
            'eval_returns_max': eval_stats.eval_returns_max,
            'eval_len_mean': eval_stats.eval_len_mean,
            'eval_len_std': eval_stats.eval_len_std,
            'eval_n_episodes': eval_stats.eval_n_episodes,
            'eval_novelty_mean': eval_stats.eval_novelty_mean,
            'eval_novelty_std': eval_stats.eval_novelty_std,
            'eval_novelty_median': eval_stats.eval_novelty_std,
            'eval_novelty_max': eval_stats.eval_novelty_max,
            'theta_norm': training_stats.theta_norm,
            'grad_norm': training_stats.grad_norm,
            'update_ratio': training_stats.update_ratio,
            'episodes_this_step': training_stats.episodes_this_step,
            'episodes_so_far': self.episodes_so_far,
            'timesteps_this_step': training_stats.timesteps_this_step,
            'timesteps_so_far': self.timesteps_so_far,
            'pop_id': pop_id,
            'overall_best_eval_returns_mean': self.best_eval_stats.eval_returns_mean,
            'overall_best_eval_returns_median': self.best_eval_stats.eval_returns_median,
            'overall_best_eval_returns_std': self.best_eval_stats.eval_returns_std,
            'overall_best_eval_returns_max': self.best_eval_stats.eval_returns_max,
            'overall_best_eval_len_mean': self.best_eval_stats.eval_len_mean,
            'overall_best_eval_len_std': self.best_eval_stats.eval_len_std,
            'overall_best_eval_novelty_mean': self.best_eval_stats.eval_novelty_mean,
            'overall_best_eval_novelty_std': self.best_eval_stats.eval_novelty_std,
            'overall_best_eval_novelty_median': self.best_eval_stats.eval_novelty_std,
            'overall_best_eval_novelty_max': self.best_eval_stats.eval_novelty_max,
            'time_elapsed_so_far': time.time() - self.t_start,
            'p_exploit': self.proba_exploit,
            'explore': self.explore,
            'nsra_weight': self.nsra_weight
        }

        log_str = '\n\t\t\tRESULTS: Population {}, \n Eval mean score {} \n ' + \
                  'Eval mean novelty {} \n Best training score {} \n Training mean score {} \n Overall best eval mean score {} \n Duration iter {}'
        logger.info(log_str.format(pop_id,
                                   eval_stats.eval_returns_mean,
                                   eval_stats.eval_novelty_mean,
                                   training_stats.po_returns_max,
                                   training_stats.po_returns_mean,
                                   self.best_eval_stats.eval_returns_mean,
                                   time.time() - t_i_step))
        self.data_logger.log(**log_data)

    def log_iteration(self, iteration, pop_id, previous_bc, previous_perf):
        if self.explore is True:
            explore = ' EXPLORE ! '
        elif self.explore is False:
            explore = ' EXPLOIT ! '
        else:
            explore = ''
        to_log = '\n\n\t\t\t'
        to_log += 'ITER {}, Algo {}.{} From pop {}. From BC {}, perf {}'.format(iteration, self.algo, explore, pop_id, previous_bc, previous_perf)
        logger.info(to_log)

    def log_obs_stats(self):
        logger.info('Diff obs mean:' + str(np.abs(self.obs_stats.mean[:5] - self.old_obs_mean[:5])))
        logger.info('Diff obs std:' + str(np.abs(self.obs_stats.std[:5] - self.old_obs_std[:5])))
        logger.info('Diff obs count:' + str(int(self.obs_stats.count - self.old_obs_count)))
        logger.info('New obs mean:' + str(self.obs_stats.mean[:5]))
        logger.info('New obs std:' + str(self.obs_stats.std[:5]))
        logger.info('New obs count:' + str(int(self.obs_stats.count)))
        self.old_obs_mean = self.obs_stats.mean
        self.old_obs_std = self.obs_stats.std
        self.old_obs_count = self.obs_stats.count

    def save_best_policy(self, policy_file, best):
        with open(policy_file, 'wt') as f:
            json.dump(best, f, sort_keys=True)
