# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:30:26 2020

@author: Han
"""
import numpy as np
import scipy.optimize as optimize
import multiprocessing as mp
import time

from models import BanditModels

def generate_kwargs(forager, opti_names, opti_value):  # Helper function for parameter intepretation
    
    if forager == 'Corrado2005':  # Special workarounds
        kwargs_all = {'forager': 'Corrado2005', 'taus': opti_value[0:2], 'w_taus': [1-opti_value[2], opti_value[2]], 'softmax_temperature': opti_value[3]}
    
    elif forager == 'Hattori2019':
        kwargs_all = {'forager': 'Hattori2019', 'step_sizes': opti_value[0:2], 'forget_rate': opti_value[2], 'softmax_temperature': opti_value[3]}

    else:
        kwargs_all = {'forager': forager}
        for (nn, vv) in zip(opti_names, opti_value):
            kwargs_all = {**kwargs_all, nn:vv}
            
    return kwargs_all


def negLL_func(fit_value, *argss):
    
    # Arguments interpretation
    forager, fit_names, choice_history, reward_history = argss
    kwargs_all = generate_kwargs(forager, fit_names, fit_value)

    # Run simulation    
    bandit = BanditModels(**kwargs_all, fit_choice_history = choice_history, fit_reward_history = reward_history)  # Into the fitting mode
    bandit.simulate()
    
    # Compute negative likelihood
    predictive_choice_prob = bandit.predictive_choice_prob  # Get all predictive choice probability [K, num_trials]
    likelihood_each_trial = predictive_choice_prob [choice_history[0,:], range(len(choice_history[0]))]  # Get the actual likelihood for each trial
    negLL = - sum(np.log(likelihood_each_trial + 1e-16))  # To avoid infinity, which makes the number of zero likelihoods informative!
    
    # print(np.round(fit_value,4), negLL, '\n')
    
    return negLL


def fit_bandit(forager, fit_names, fit_bounds, choice_history, reward_history):
    # now = time.time()
    # n_worker = 1
    n_worker = int(mp.cpu_count())   
    if n_worker > 1: 
        updating='deferred' 
    else: 
        updating='immediate'
                
    # Parameter optimization with DE    
    fitting_result = optimize.differential_evolution(func = negLL_func, args = (forager, fit_names, choice_history, reward_history),
                                                     bounds = optimize.Bounds(fit_bounds[0], fit_bounds[1]), 
                                                     mutation=(0.5, 1), recombination = 0.7, popsize = 16,
                                                     workers = n_worker, disp = n_worker==1, strategy = 'best1bin', updating=updating)

    # print(fitting_result, 'time = %g' % (time.time() - now))
    return fitting_result