"""
    Add the environment you wish to train here
"""

import gym
import numpy as np
from rllab.envs.gym_env import GymEnv

# ===================================================================
# Functions used in training and testing phase

def make_train_MDP():
    return _standard_half_cheetah()


def make_test_MDP(param=None):
    return _heavy_half_cheetah(param)


def make_custom_MDP():
    return _ensemble_hopper()

# ===================================================================
# Local functions to create envs

def _ensemble_hopper():
    return GymEnv("UncertainHopper-v0")

def _standard_walker():
    return GymEnv('Walker2d-v1')

def _heavy_walker():
    # Make the torso heavy
    e = GymEnv('Walker2d-v1')
    bm = np.array(e.env.model.body_mass)
    gs = np.array(e.env.model.geom_size)
    bm[1] = 7; gs[1][0] = 0.1;
    e.env.model.body_mass = bm; e.env.model.geom_size = gs;
    return e

def _standard_half_cheetah():
    return GymEnv("HalfCheetah-v1")

def _heavy_half_cheetah(mass=None):
    #Make the torso heavy, default mass is 6.3
    e = GymEnv("HalfCheetah-v1")
    bm = np.array(e.env.model.body_mass)
    gs = np.array(e.env.model.geom_size)
    if mass == None:
        bm[1] = 14
    else:
        bm[1] = mass
    gs[1][0] = (0.05/3.5343)*bm[1];
    e.env.model.body_mass = bm; e.env.model.geom_size = gs;
    return e
# =======================================================================================
# Generate environment corresponding to the given mode

def get_environment(env_mode, param=None):

    modes = ['train', 'test', 'custom', 'standard', 'heavy', 'random']    

    if env_mode == 'train':
        env = make_train_MDP()
    elif env_mode == 'test':
        env = make_test_MDP(param)
    elif env_mode == 'custom':
        env = make_custom_MDP()
    elif env_mode == 'standard':
        env = _make_standard_MDP()
    elif env_mode == 'heavy':
        env = _make_heavy_MDP()
    elif env_mode == 'random':
        env = _make_random_MDP()
    else:
        print "ERROR: Unknown environment mode specified. Allowed modes are ", modes

    return env
