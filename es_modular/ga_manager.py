# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time

import numpy as np
import json

from es_modular.ga import GAIndividual, initialize_worker_fiber 
from es_modular.bc_archive import BCArchive
from es_modular.logger import CSVLogger, log_fields
from es_modular.stats import RunningStat
from es_modular.interaction.interaction import build_interaction_module

logger = logging.getLogger(__name__)

try:
    import fiber as mp
    logger.info('Using Fiber')
except:
    import multiprocessing as mp
    logger.info('Using Multiprocessing')


class GAPopulationManager:
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
        logger.info(('Worker pool created'))

        self.algo = args['algo']
        self.use_norm_obs = args['env_args']['use_norm_obs']
        self.population = dict()
        self.env_id = args['env_id']
        self.noise_std = self.args['noise_std']

        self.observation_shape = self.interaction.env.observation_space.shape

        self.best = dict(performance=-1e6)
        self.best_eval_stats = None

        args['env_args'].update(nb_cells_per_dimension=self.interaction.nb_cells_per_dimension,
                                min_max_bcs=self.interaction.min_max_bcs,
                                dim_bc=self.interaction.dim_bc)
        self.bc_archive = BCArchive(args)

        theta, obs_mean, obs_std = None, None, None

        # create initial population
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

        self.explore = True

        # logging
        log_path = args['log_dir'] + '/' + args['log_dir'].split('/')[-1] + 'results.csv'
        self.data_logger = CSVLogger(log_path, log_fields)
        self.filename_best = args['log_dir'] + '/' + args['log_dir'].split('/')[-1] + 'best_policy.json'
        self.t_start = time.time()
        self.episodes_so_far = 0
        self.timesteps_so_far = 0

    def add_individual(self, theta=None, perf=None, bc=None, pop_id=None):
        # population is a dict:
        # For mega, pop_id is -1.
        if theta is None:
            theta = self.interaction.initial_theta()

        self.population[pop_id] = GAIndividual(pop_id=pop_id,
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

        population = self.population[pop_id]
        assert population.pop_id == pop_id

        # # # # # #
        # Training
        # # # # # #

        # syncrhonize obs stats
        if self.use_norm_obs:
            population.update_obs_stats(self.obs_stats)

        # sample pseudo-offsprings and evaluate them
        thetas, training_task = population.start_step(population.theta)

        # compute update for new parent from offsprings
        perfs, novs, bcs, final_xpos, obs_updates, stats, training_stats = population.get_step(training_task=training_task)

        self.update_best(pop_id, iteration, perfs, thetas, bcs, stats)

        # # # # # # # # #
        # Update Archive
        # # # # # # # # #

        # update archive
        inds = np.flip(np.argsort(perfs).flatten(), axis=0)
        for i in inds:
            new_bc = bcs[i]
            new_perf = perfs[i]
            new_novelty = novs[i]
            self.bc_archive.add(new_bc=new_bc.copy(),
                                performance=new_perf,
                                best_train_performance=training_stats.po_returns_max,
                                previous_bc=previous_bc.copy(),
                                novelty=new_novelty,
                                theta=thetas[i],
                                iter=iteration,
                                explore=None,
                                obs_stats=self.obs_stats,
                                starting_bc=starting_bc,
                                pop_id=pop_id,
                                final_xpos=final_xpos[i])
        i_best = 0
        best_perf = stats[0].eval_returns_mean
        for i in range(len(stats)):
            if stats[i].eval_returns_mean > best_perf:
                i_best = i
                best_perf = stats[i].eval_returns_mean

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
                              eval_stats=stats[i_best],
                              t_i_step=t_i_step,
                              pop_id=pop_id
                              )

        return new_bc, new_perf

    def optimize(self, iterations):

        start = True
        iteration = -2
        while iteration < iterations:

            logger.info('\t\tSAMPLING NEW STARTING POLICY\n')

            self.explore = not self.explore
            # decide which theta to evolve next
            if not start:
                starting_pop_id, starting_theta, starting_bc, starting_perf = self.bc_archive.sample(self.explore)
                if starting_pop_id in self.population.keys():
                    del self.population[starting_pop_id]
                self.add_individual(theta=starting_theta, bc=previous_bc.copy(), perf=starting_perf.copy(), pop_id=starting_pop_id)
                assert starting_theta[0] == self.population[starting_pop_id].theta[0]
            else:
                start = False
                starting_pop_id = -1
                starting_bc = np.zeros([2])
                starting_perf = 0

            previous_bc = starting_bc.copy()
            previous_perf = starting_perf
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


    def update_best(self, pop_id, iteration, perfs, thetas, bcs, stats):
        for i in range(len(perfs)):
            if perfs[i] > self.best['performance']:
                self.best.update(theta=thetas[i].tolist(),
                                 performance=perfs[i],
                                 bc=bcs[i].tolist(),
                                 iter=iteration)
                if self.use_norm_obs:
                    self.best.update(obs_mean=self.obs_stats.mean.tolist(),
                                     obs_std=self.obs_stats.std.tolist())
                self.best_eval_stats = stats[i]
                logger.info('New best score: {}, BC {}, from population {}'.format(perfs[i],
                                                                                   bcs[i],
                                                                                   pop_id))



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
            'p_exploit': None,
            'explore': None,
            'nsra_weight': None
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
