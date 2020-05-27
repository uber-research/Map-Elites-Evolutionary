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
from es_modular.stats import batched_weighted_sum
from es_modular.logger import StepStats, EvalStats, POResult
from .optimizers import Adam, SimpleSGD
from es_modular.interaction.interaction import build_interaction_module

logger = logging.getLogger(__name__)


def initialize_worker_fiber(arg_thetas, args, env_id, arg_obs_mean, arg_obs_std):
    global noise, theta_shared, interaction_shared, obs_mean_shared, obs_std_shared
    from .noise_module import noise
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


class GAIndividual:
    def __init__(self, pop_id, algo, args, fiber_pool, fiber_shared, theta,
                 bc, perf, bc_archive):

        logger.info('Creating population {}...'.format(pop_id))
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

        self.use_norm_obs = self.args['env_args']['use_norm_obs']

        # obs_stats
        self.obs_mean = None
        self.obs_std = None

        self.noise_std = self.args['noise_std']
        self.nb_evals = self.args['eval_batches_per_step']

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

    def start_chunk(self, runner, batch_size, noise_theta, noise_std):
        chunk_tasks = []
        pool = self.fiber_pool
        for i in range(self.nb_evals // batch_size):
            chunk_tasks.append(pool.apply_async(runner, args=(batch_size,
                                                              noise_theta,
                                                              noise_std)))
        return chunk_tasks


    def get_chunk(self, tasks):
        return [task.get() for task in tasks]

    def collect_po_results(self, po_results, per_batch):
        returns = np.concatenate([r.returns for r in po_results])
        lengths = np.concatenate([r.lengths for r in po_results])
        bcs = np.concatenate([r.bcs.T for r in po_results])
        final_xposs = np.concatenate([r.final_xpos for r in po_results])

        all_returns = []
        all_lengths = []
        all_bcs = []
        all_final_xpos = []
        for i in range(len(returns) // per_batch):
            inds = np.arange(i * per_batch, (i + 1) * per_batch)
            all_returns.append(returns[inds])
            all_bcs.append(bcs[inds])
            all_lengths.append(lengths[inds])
            all_final_xpos.append(final_xposs[inds])

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
        return all_returns, all_lengths, all_bcs, all_final_xpos, obs_updates, returns, lengths, bcs, final_xposs

    def compute_grads(self, noise_inds, fitness, theta):
        grads, count = batched_weighted_sum(fitness[:, 0] - fitness[:, 1],
                                            (noise.get(idx, len(theta)) for idx in noise_inds), batch_size=500)
        grads /= len(fitness)
        if self.args['optimizer_args']['divide_gradient_by_noise_std']:
            grads /= self.noise_std
        return grads

    def update_obs_stats(self, obs_stats):
        self.obs_mean = obs_stats.mean
        self.obs_std = obs_stats.std

    def start_step(self, theta):
        global noise

        self.broadcast_theta(theta)

        rs_seed = np.random.randint(np.int32(2 ** 31 - 1))
        random_state = np.random.RandomState(rs_seed)
        random_state.seed(rs_seed)
        n_thetas = self.batch_size * self.batches_per_step * 2 // self.nb_evals + 1
        noise_inds = np.asarray([noise.sample_index(random_state, len(theta)) for _ in range(n_thetas)], dtype='int')

        self.broadcast_obs_stats(self.obs_mean, self.obs_std)

        thetas = [theta + self.noise_std * noise.get(noise_id, len(theta)) for noise_id in noise_inds]
        training_task = []
        for i in range(n_thetas):
            training_task += self.start_chunk(run_po_rollout_batch,
                                              self.batch_size,
                                              noise_inds[i],
                                              self.noise_std)
        return thetas, training_task

    def get_step(self, training_task):
        step_results = self.get_chunk(training_task)

        batch_returns, batch_lengths, batch_bcs, batch_final_xpos, obs_updates, \
        po_returns, po_lengths, po_bcs, po_final_xposs = self.collect_po_results(step_results, per_batch=self.nb_evals)

        episodes_this_step = po_returns.size
        timesteps_this_step = po_lengths.sum()
        logger.info('Population {} finished running {} episodes, {} timesteps.'.format(
            self.pop_id, episodes_this_step, timesteps_this_step))

        n_thetas = len(batch_returns)
        # compute fitness
        po_novelty = self.compute_novelty(po_bcs)

        novelties = []
        performances = []
        final_xpos = []
        av_bcs = []
        for i in range(n_thetas):
            novelties.append(self.compute_novelty(batch_bcs[i]))
            performances.append(batch_returns[i].mean())
            final_xpos.append(batch_final_xpos[i].mean())
            av_bcs.append(batch_bcs[i].mean(axis=0))

        stats = []
        for i in range(n_thetas):
            stats.append(EvalStats(eval_returns_mean=batch_returns[i].mean(),
                                   eval_returns_median=np.median(batch_returns[i]),
                                   eval_returns_std=batch_returns[i].std(),
                                   eval_returns_max=batch_returns[i].max(),
                                   eval_len_mean=batch_lengths[i].mean(),
                                   eval_len_std=batch_lengths[i].std(),
                                   eval_n_episodes=len(batch_returns[i]) * len(batch_returns),
                                   eval_novelty_mean=novelties[i].mean(),
                                   eval_novelty_median=np.median(novelties[i]),
                                   eval_novelty_std=novelties[i].std(),
                                   eval_novelty_max=novelties[i].max()
                                   ))

        for i in range(n_thetas):
            novelties[i] = novelties[i].mean()

        return performances, novelties, av_bcs, final_xpos, obs_updates, stats, StepStats(po_returns_mean=po_returns.mean(),
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
                                                                                          learning_rate=0,
                                                                                          theta_norm=0,
                                                                                          grad_norm=0,
                                                                                          update_ratio=0,
                                                                                          episodes_this_step=episodes_this_step,
                                                                                          timesteps_this_step=timesteps_this_step,
                                                                                          )

    def compute_novelty(self, bcs):
        shape = bcs.shape
        dim_bc = shape[-1]
        bcs = bcs.reshape([np.prod(shape[:-1]), dim_bc])
        av_distance_to_knn = self.bc_archive.compute_novelty(bcs)
        novelty = av_distance_to_knn.reshape(shape[:-1])
        return novelty

    def set_noise_std(self, noise_std):
        self.noise_std = noise_std


# rollout function
def run_po_rollout_batch(batch_size, noise_theta, noise_std=None):
    global noise
    t_init = time.time()
    interaction = interaction_shared
    theta = fiber_get_theta()
    obs_mean, obs_std = fiber_get_obs_stats()

    assert noise_std is not None

    random_state = np.random.RandomState()
    thetas = (theta + noise_std * noise.get(noise_theta, len(theta)) for _ in range(batch_size))
    returns, lengths, bcs, final_xpos, obs_sum, obs_sq, obs_count = interaction.rollout_batch(thetas=thetas,
                                                                                              batch_size=batch_size,
                                                                                              random_state=random_state,
                                                                                              obs_mean=obs_mean,
                                                                                              obs_std=obs_std)

    end = time.time() - t_init
    return POResult(returns=returns, noise_inds=noise_theta, lengths=lengths, bcs=np.swapaxes(np.array(bcs), 0, 1),
                    obs_sum=obs_sum, obs_sq=obs_sq, obs_count=obs_count, time=end, final_xpos=final_xpos)
