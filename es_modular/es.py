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

from es_modular.noise_module import noise
from es_modular.stats import compute_centered_ranks, batched_weighted_sum
from es_modular.logger import StepStats, EvalStats, EvalResult, POResult
from es_modular.optimizers import Adam, SimpleSGD
from es_modular.interaction.interaction import build_interaction_module

logger = logging.getLogger(__name__)


def initialize_worker_fiber(arg_thetas, args, env_id, arg_obs_mean, arg_obs_std):
    global noise, theta_shared, interaction_shared, obs_mean_shared, obs_std_shared
    theta_shared = arg_thetas
    interaction_shared = build_interaction_module(env_id=env_id, args=args)
    obs_mean_shared = arg_obs_mean
    obs_std_shared = arg_obs_std


def fiber_get_theta():
    return theta_shared[0]


def fiber_get_obs_stats():
    return obs_mean_shared[0], obs_std_shared[0]


def get_optim(args):
    if args['optimizer_args']['optimizer'] == 'sgd':
        return SimpleSGD
    elif args['optimizer_args']['optimizer'] == 'adam':
        return Adam
    else:
        raise NotImplementedError


class ESIndividual:
    def __init__(self, pop_id, algo, args, fiber_pool, fiber_shared, theta,
                 bc, perf, bc_archive):

        self.args = args
        self.algo = algo
        self.pop_id = pop_id
        self.theta = theta
        self.bc = bc
        self.perf = perf
        self.bc_archive = bc_archive

        self.batches_per_step = self.args['num_workers']
        self.batch_size = self.args['batch_size']
        self.eval_batch_size = self.args['eval_batch_size']
        self.eval_batches_per_step = self.args['eval_batches_per_step']
        self.l2_coeff = self.args['optimizer_args']['l2_coeff']
        self.fitness_normalization = self.args['fitness_normalization']

        # fiber
        self.fiber_pool = fiber_pool
        self.fiber_shared = fiber_shared

        logger.info('Population {} optimizing {} parameters'.format(pop_id, len(theta)))

        self.optimizer = get_optim(self.args)(self.theta, stepsize=self.args['optimizer_args']['learning_rate'])

        self.use_norm_obs = self.args['env_args']['use_norm_obs']

        # obs_stats
        self.obs_mean = None
        self.obs_std = None

        self.noise_std = self.args['noise_std']
        self.learning_rate = self.args['optimizer_args']['learning_rate']

        logger.info('Population {} created!'.format(pop_id))

    def broadcast_theta(self, theta):
        '''On all worker, set thetas[this population] to theta'''
        theta_shared = self.fiber_shared["theta"]
        theta_shared[0] = theta

    def broadcast_obs_stats(self, obs_mean, obs_std):
        '''On all worker, set obs_means[this population] to obs_mean etc'''
        obs_mean_shared = self.fiber_shared["obs_mean"]
        obs_mean_shared[0] = obs_mean
        obs_std_shared = self.fiber_shared["obs_std"]
        obs_std_shared[0] = obs_std

    def start_chunk(self, runner, batches_per_step, batch_size, *args):

        rs_seeds = np.random.randint(np.int32(2 ** 31 - 1), size=batches_per_step)

        chunk_tasks = []
        pool = self.fiber_pool

        for i in range(batches_per_step):
            chunk_tasks.append(pool.apply_async(runner, args=(batch_size,
                                                              rs_seeds[i]) + args))
        return chunk_tasks

    def get_chunk(self, tasks):
        return [task.get() for task in tasks]

    def collect_po_results(self, po_results):
        noise_inds = np.concatenate([r.noise_inds for r in po_results])
        returns = np.concatenate([r.returns for r in po_results])
        lengths = np.concatenate([r.lengths for r in po_results])
        bcs = np.concatenate([r.bcs for r in po_results])

        if not self.use_norm_obs:
            obs_updates = dict(obs_sums=None,
                               obs_sqs=None,
                               obs_counts=None)
        else:
            obs_sums = np.concatenate([r.obs_sum.reshape(1, -1) for r in po_results], axis=0)
            obs_sqs = np.concatenate([r.obs_sq.reshape(1, -1) for r in po_results], axis=0)
            obs_counts = np.array([r.obs_count for r in po_results])
            obs_updates = dict(obs_sums=obs_sums,
                               obs_sqs=obs_sqs,
                               obs_counts=obs_counts)
        return noise_inds, returns, lengths, bcs, obs_updates

    def collect_eval_results(self, eval_results):
        eval_returns = np.concatenate([r.returns for r in eval_results])
        eval_lengths = np.concatenate([r.lengths for r in eval_results])
        eval_bcs = np.concatenate([r.bcs for r in eval_results])
        eval_final_xpos = np.concatenate([r.final_xpos for r in eval_results])
        return eval_returns, eval_lengths, eval_bcs, eval_final_xpos

    def compute_grads(self, noise_inds, fitness, theta):
        grads, count = batched_weighted_sum(fitness[:, 0] - fitness[:, 1],
                                            (noise.get(idx, len(theta)) for idx in noise_inds), batch_size=500)
        grads /= len(fitness)
        if self.args['optimizer_args']['divide_gradient_by_noise_std']:
            grads /= self.noise_std
        return grads

    def start_theta_eval(self, theta):
        logger.info('Starting evaluation of population {}.'.format(self.pop_id))
        self.broadcast_theta(theta)
        self.broadcast_obs_stats(self.obs_mean, self.obs_std)

        eval_task = self.start_chunk(run_eval_rollout_batch,
                                     self.eval_batches_per_step,
                                     self.eval_batch_size,
                                     )
        return eval_task

    def get_theta_eval(self, new_theta, eval_task, start=False):
        eval_results = self.get_chunk(eval_task)
        eval_returns, eval_lengths, eval_bcs, eval_final_xpos = self.collect_eval_results(eval_results)

        if start:
            eval_novelties = 10 * np.ones([3])  # optimistic evaluation to run each population once at the beginning
        else:
            eval_novelties = self.compute_novelty(eval_bcs)

        logger.info('Evaluation of best theta finished running {} episodes.'.format(len(eval_returns)))

        self.theta = new_theta.copy()
        self.perf = eval_returns.mean()
        self.bc = eval_bcs.mean(axis=0)

        return eval_bcs, eval_returns, eval_novelties, eval_final_xpos, EvalStats(eval_returns_mean=eval_returns.mean(),
                                                                                  eval_returns_median=np.median(eval_returns),
                                                                                  eval_returns_std=eval_returns.std(),
                                                                                  eval_returns_max=eval_returns.max(),
                                                                                  eval_len_mean=eval_lengths.mean(),
                                                                                  eval_len_std=eval_lengths.std(),
                                                                                  eval_n_episodes=len(eval_returns) * len(eval_returns),
                                                                                  eval_novelty_mean=eval_novelties.mean(),
                                                                                  eval_novelty_median=np.median(eval_novelties),
                                                                                  eval_novelty_std=eval_novelties.std(),
                                                                                  eval_novelty_max=eval_novelties.max()
                                                                                  )

    def update_obs_stats(self, obs_stats):
        self.obs_mean = obs_stats.mean
        self.obs_std = obs_stats.std

    def start_step(self):
        self.broadcast_theta(self.theta)
        self.broadcast_obs_stats(self.obs_mean, self.obs_std)

        training_task = self.start_chunk(run_po_rollout_batch,
                                         self.batches_per_step,
                                         self.batch_size,
                                         self.noise_std)
        return training_task

    def get_step(self, training_task, explore, nsra_weight):
        step_results = self.get_chunk(training_task)

        noise_inds, po_returns, po_lengths, po_bcs, obs_updates = self.collect_po_results(step_results)

        episodes_this_step = po_returns.size
        timesteps_this_step = po_lengths.sum()
        logger.info('Population {} finished running {} episodes, {} timesteps.'.format(
            self.pop_id, episodes_this_step, timesteps_this_step))

        # compute fitness
        po_proc_fitnesses, po_novelty = self.compute_fitness(returns=po_returns, bcs=po_bcs, explore=explore, nsra_weight=nsra_weight)

        # compute grads and update optimizer
        current_theta = self.theta.copy()
        grad = self.compute_grads(noise_inds, po_proc_fitnesses, current_theta)
        update_ratio, updated_theta = self.optimizer.update(current_theta, - grad + self.l2_coeff * current_theta)

        return updated_theta.copy(), self.optimizer, obs_updates, StepStats(po_returns_mean=po_returns.mean(),
                                                                            po_returns_median=np.median(po_returns),
                                                                            po_returns_std=po_returns.std(),
                                                                            po_returns_max=po_returns.max(),
                                                                            po_novelty_mean=po_novelty.mean(),
                                                                            po_novelty_std=po_novelty.std(),
                                                                            po_novelty_median=np.median(po_novelty),
                                                                            po_novelty_max=po_novelty.max(),
                                                                            po_returns_min=po_returns.min(),
                                                                            po_len_mean=po_lengths.mean(),
                                                                            po_len_std=po_lengths.std(),
                                                                            po_len_max=po_lengths.max(),
                                                                            noise_std=self.noise_std,
                                                                            learning_rate=self.optimizer.stepsize,
                                                                            theta_norm=np.square(self.theta).sum(),
                                                                            grad_norm=float(np.square(grad).sum()),
                                                                            update_ratio=float(update_ratio),
                                                                            episodes_this_step=episodes_this_step,
                                                                            timesteps_this_step=timesteps_this_step,
                                                                            )

    def compute_novelty(self, bcs):
        # compute novelty w.r.t. all theta previously encountered
        shape = bcs.shape
        dim_bc = shape[-1]
        bcs = bcs.reshape([np.prod(shape[:-1]), dim_bc])
        av_distance_to_knn = self.bc_archive.compute_novelty(bcs)
        novelty = av_distance_to_knn.reshape(shape[:-1])
        return novelty

    def compute_fitness(self, returns, bcs, explore, nsra_weight=0.5):
        novelty = self.compute_novelty(bcs)
        if self.algo == 'nses' or ('mees' in self.algo and explore):
            proc_fitness = self.normalize_fitness(novelty)
        elif self.algo == 'nsres':
            proc_fitness = (self.normalize_fitness(novelty) + self.normalize_fitness(returns)) / 2.0
        elif self.algo == 'nsraes':
            proc_fitness = (1 - nsra_weight) * self.normalize_fitness(novelty) + nsra_weight * self.normalize_fitness(returns)
        elif 'mees' in self.algo and not explore:
            proc_fitness = self.normalize_fitness(returns)
        else:
            raise NotImplementedError
        return proc_fitness, novelty

    def normalize_fitness(self, fitness):
        if self.fitness_normalization == 'centered_ranks':
            proc_fitness = compute_centered_ranks(fitness)
        elif self.fitness_normalization == 'normal':
            proc_fitness = (fitness - fitness.mean()) / (fitness.std() + 1e-5)
        else:
            raise NotImplementedError('Invalid return normalization `{}`'.format(self.fitness_normalization))
        return proc_fitness


# rollout function
def run_po_rollout_batch(batch_size, rs_seed, noise_std=None):
    global noise
    t_init = time.time()
    interaction = interaction_shared
    theta = fiber_get_theta()
    obs_mean, obs_std = fiber_get_obs_stats()
    random_state = np.random.RandomState(rs_seed)
    random_state.seed(rs_seed)

    assert noise_std is not None
    noise_inds = np.asarray([noise.sample_index(random_state, len(theta)) for _ in range(batch_size)], dtype='int')

    returns = np.zeros((batch_size, 2))
    final_xpos = np.zeros((batch_size, 2))
    lengths = np.zeros((batch_size, 2), dtype='int')
    bcs = [None] * 2

    # mirror sampling
    thetas = (theta + noise_std * noise.get(noise_idx, len(theta)) for noise_idx in noise_inds)
    returns[:, 0], lengths[:, 0], bcs[0], final_xpos[:, 0], _, _, _, = interaction.rollout_batch(thetas=thetas,
                                                                                                 batch_size=batch_size,
                                                                                                 random_state=random_state,
                                                                                                 obs_mean=obs_mean,
                                                                                                 obs_std=obs_std)
    thetas = (theta - noise_std * noise.get(noise_idx, len(theta)) for noise_idx in noise_inds)
    returns[:, 1], lengths[:, 1], bcs[1], final_xpos[:, 1], obs_sum, obs_sq, obs_count = interaction.rollout_batch(thetas=thetas,
                                                                                                                   batch_size=batch_size,
                                                                                                                   random_state=random_state,
                                                                                                                   obs_mean=obs_mean,
                                                                                                                   obs_std=obs_std)
    end = time.time() - t_init
    return POResult(returns=returns, noise_inds=noise_inds, lengths=lengths, bcs=np.swapaxes(np.array(bcs), 0, 1),
                    obs_sum=obs_sum, obs_sq=obs_sq, obs_count=obs_count, time=end, final_xpos=final_xpos)


def run_eval_rollout_batch(batch_size, rs_seed):
    global noise
    interaction = interaction_shared
    theta = fiber_get_theta().copy()
    obs_mean, obs_std = fiber_get_obs_stats()
    random_state = np.random.RandomState(rs_seed)
    random_state.seed(rs_seed)
    returns, lengths, bcs, final_xpos, _, _, _ = interaction.rollout_batch(thetas=(theta for _ in range(batch_size)),
                                                                           batch_size=batch_size,
                                                                           random_state=random_state,
                                                                           eval=True,
                                                                           obs_mean=obs_mean,
                                                                           obs_std=obs_std)
    return EvalResult(returns=returns, lengths=lengths, bcs=np.array(bcs).reshape(1, -1), final_xpos=final_xpos)
