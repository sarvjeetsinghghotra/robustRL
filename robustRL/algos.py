"""
    This file contains implimentations of various batch policy optimization
    algorithms. See rllab documentation for more details about batchpolopt.
    Currently, we have implimentations for TRPO and REINFORCE

    Aravind Rajeswaran, 08/04/16
"""

from __future__ import print_function

import logging
logging.disable(logging.CRITICAL)
import sys
sys.dont_write_bytecode = True

import numpy as np
import theano
#theano.sandbox.cuda.unuse()
import theano.tensor as TT
from lasagne.updates import adam
import pickle
import copy
import time as timer
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.misc import ext
from rllab.misc import tensor_utils
from rllab.misc.ext import set_seed as rllab_set_seed

from robustRL.samplers import *
import robustRL.utils



class TRPO:
   
    def __init__(self, env, policy, baseline, max_kl):
        """ env     = only structural info of env is used here; 
                      you need to pass the 'mode' to functions of this class
            max_kl  = constraint for determining step-size (suggested: 1e-2 or 5e-3)
        """
        
        self.policy     = policy
        self.env        = env
        self.baseline   = baseline

        self.optimizer  = ConjugateGradientOptimizer(**dict())

        # Define symbolic variables
        self.observations_var = self.env.observation_space.new_tensor_variable('observations', extra_dims=1)
        self.actions_var      = self.env.action_space.new_tensor_variable('actions', extra_dims=1)
        self.advantages_var   = TT.vector('advantages')

        self.dist = self.policy.distribution  

        self.old_dist_info_vars = {
            k: ext.new_tensor(
                'old_%s' % k,
                ndim=2,
                dtype=theano.config.floatX
            ) for k in self.dist.dist_info_keys
            }
        self.old_dist_info_vars_list = [self.old_dist_info_vars[k] for k in self.dist.dist_info_keys]

        self.state_info_vars = {
            k: ext.new_tensor(
                k,
                ndim=2,
                dtype=theano.config.floatX
            ) for k in self.policy.state_info_keys
        }
        self.state_info_vars_list = [self.state_info_vars[k] for k in self.policy.state_info_keys]

        self.dist_info_vars = self.policy.dist_info_sym(self.observations_var, self.state_info_vars)   
        # distribution info variable (symbolic) -- interpret as pi
        self.KL = self.dist.kl_sym(self.old_dist_info_vars, self.dist_info_vars)
        self.LR = self.dist.likelihood_ratio_sym(self.actions_var, self.old_dist_info_vars, self.dist_info_vars)
        self.mean_KL = TT.mean(self.KL)
        
        self.surr = - TT.mean(self.LR * self.advantages_var)

        self.input_list = [self.observations_var, self.actions_var, self.advantages_var] + \
                          self.state_info_vars_list + self.old_dist_info_vars_list
        self.optimizer.update_opt(loss=self.surr, target=self.policy, \
                                  leq_constraint=(self.mean_KL, max_kl), \
                                  inputs=self.input_list, constraint_name="mean_kl")


    def train(self, N, T, gamma, niter, env_mode='train'):
        """    N = number of trajectories
               T = horizon
               niter = number of iterations to update the policy
               env_mode = can be 'train', 'test' or something else. 
                          You need to write the appropriate function in MDP_funcs
        """        

        eval_statistics = []
        for iter in range(niter):
            curr_iter_stats = self.train_step(N, T, gamma, env_mode)
            eval_statistics.append(curr_iter_stats)

        return eval_statistics


    def train_step(self, N, T, gamma, env_mode='train', 
        num_cpu='max',
        save_paths=False,
        idx=None, 
        normalized_env=False):
        """    N = number of trajectories
               T = horizon
               env_mode = can be 'train', 'test' or something else. 
                          You need to write the appropriate function in MDP_funcs
        """
        
        paths = sample_paths_parallel(N, self.policy, self.baseline, 
            env_mode, T, gamma, num_cpu=num_cpu, normalized_env=normalized_env)

        # save the paths used to make the policy update
        if save_paths == True and idx != None:
            robustRL.utils.save_paths(paths, idx)

        eval_statistics = self.train_from_paths(paths)
        eval_statistics.append(N)

        return eval_statistics


    def train_from_paths(self, paths):
        
        self.baseline.fit(paths)
        # concatenate from all the trajectories
        observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
        actions      = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
        rewards      = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
        advantages   = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
        env_infos    = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
        agent_infos  = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

        samples_data = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            advantages=advantages,
            env_infos=env_infos,
            agent_infos=agent_infos,
        )
        
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        
        # Take a step with optimizer
        self.optimizer.optimize(all_input_values)

        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return  = np.mean(path_returns)
        std_return   = np.std(path_returns)
        min_return   = np.amin(path_returns)
        max_return   = np.amax(path_returns)
        return [mean_return, std_return, min_return, max_return]



class REINFORCE:
   
    def __init__(self, env, policy, baseline, max_kl):
        """ env     = only structural info of env is used here; 
                      you need to pass the 'mode' to functions of this class
            max_kl  = constraint for determining step-size (suggested: 1e-2 or 5e-3)
        """
        
        self.policy     = policy
        self.env        = env
        self.baseline   = baseline

        self.optimizer  = FirstOrderOptimizer(**dict())

        # Define symbolic variables
        self.observations_var = self.env.observation_space.new_tensor_variable('observations', extra_dims=1)
        self.actions_var      = self.env.action_space.new_tensor_variable('actions', extra_dims=1)
        self.advantages_var   = TT.vector('advantages')

        self.dist = self.policy.distribution  

        self.old_dist_info_vars = {
            k: ext.new_tensor(
                'old_%s' % k,
                ndim=2,
                dtype=theano.config.floatX
            ) for k in self.dist.dist_info_keys
            }
        self.old_dist_info_vars_list = [self.old_dist_info_vars[k] for k in self.dist.dist_info_keys]

        self.state_info_vars = {
            k: ext.new_tensor(
                k,
                ndim=2,
                dtype=theano.config.floatX
            ) for k in self.policy.state_info_keys
        }
        self.state_info_vars_list = [self.state_info_vars[k] for k in self.policy.state_info_keys]

        self.dist_info_vars = self.policy.dist_info_sym(self.observations_var, self.state_info_vars)
        self.logli = self.dist.log_likelihood_sym(self.actions_var, self.dist_info_vars)

        self.surr = - TT.mean(logli * advantages_var)

        self.input_list = [self.observations_var, self.actions_var, self.advantages_var] + self.state_info_vars_list
        self.optimizer.update_opt(self.surr, target=self.policy, inputs=input_list)


    def train(self, N, T, gamma, niter, env_mode='train'):
        """    N = number of trajectories
               T = horizon
               niter = number of iterations to update the policy
               env_mode = can be 'train', 'test' or something else. 
                          You need to write the appropriate function in MDP_funcs
        """        

        eval_statistics = []
        for iter in range(niter):
            curr_iter_stats = self.train_step(N, T, gamma, env_mode)
            eval_statistics.append(curr_iter_stats)

        return eval_statistics
        

    def train_step(self, N, T, gamma, env_mode='train'):
        """    N = number of trajectories
               T = horizon
               env_mode = can be 'train', 'test' or something else. 
                          You need to write the appropriate function in MDP_funcs
        """
        
        paths = sample_paths_parallel(N, T, gamma, self.policy, self.baseline, env, num_cpu='max')

        eval_statistics = self.train_from_paths(paths)

        return eval_statistics


    def train_from_paths(self, paths):
        
        self.baseline.fit(paths)
        # concatenate from all the trajectories
        observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
        actions      = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
        rewards      = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
        advantages   = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
        env_infos    = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
        agent_infos  = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

        samples_data = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            advantages=advantages,
            env_infos=env_infos,
            agent_infos=agent_infos,
        )
        
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list)
        
        # Take a step with optimizer
        self.optimizer.optimize(all_input_values)

        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return  = np.mean(path_returns)
        std_return   = np.std(path_returns)
        min_return   = np.amin(path_returns)
        max_return   = np.amax(path_returns)
        return (mean_return, std_return, min_return, max_return)