# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:30:26 2020

@author: Han
"""
import numpy as np
import scipy.optimize as optimize
import multiprocessing as mp
from tqdm import tqdm  # For progress bar. HH
import time

from models import BanditModels
global fit_history

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

def callback_DE(x, **kargs):
    global fit_history
    fit_history.append(x)
    
    return

def fit_each_init(forager, fit_names, fit_bounds, choice_history, reward_history, fit_method):
    x0 = []
    for lb,ub in zip(fit_bounds[0], fit_bounds[1]):
        x0.append(np.random.uniform(lb,ub))
        
    fitting_result = optimize.minimize(negLL_func, x0, args = (forager, fit_names, choice_history, reward_history), method = fit_method,
                                       bounds = optimize.Bounds(fit_bounds[0], fit_bounds[1]), 
                                       )
    return fitting_result


def fit_bandit(forager, fit_names, fit_bounds, choice_history, reward_history, if_callback = False, fit_method = 'CG', n_x0s = 1, pool = ''):
    # now = time.time()
    
    # -- For DE, use pool to control if_parallel, although we don't use pool for DE.
    if pool != '':
        n_worker = int(mp.cpu_count()/2)    
        updating='deferred' 
    else:
        n_worker = 1
        updating='immediate'
        
    if if_callback:  # Store the intermediate DE results
        global fit_history
        callback = callback_DE
    else:
        callback = None
                
    # -- Parameter optimization --
    fit_history = []
    
    if fit_method == 'DE':
        # Use DE's own parallel method
        fitting_result = optimize.differential_evolution(func = negLL_func, args = (forager, fit_names, choice_history, reward_history),
                                                         bounds = optimize.Bounds(fit_bounds[0], fit_bounds[1]), 
                                                         mutation=(0.5, 1), recombination = 0.7, popsize = 16, 
                                                         workers = n_worker, disp = False, strategy = 'best1bin', 
                                                         updating=updating, callback = callback,)
        
    elif fit_method in ['L-BFGS-B', 'SLSQP', 'TNC', 'trust-constr']:
        
        # Do parallel initialization
        fitting_parallel_results = []
        
        if pool != '':  # Go parallel
            results = []
            
            '''
            Must use two separate for loops, one for assigning and one for harvesting!
            '''
            for nn in range(n_x0s):
                # Assign jobs
                results.append(pool.apply_async(fit_each_init, args = (forager, fit_names, fit_bounds, choice_history, reward_history, fit_method)))
                
            for rr in results:
                # Get data    
                fitting_parallel_results.append(rr.get())
        else:
            # Serial
            result = fit_each_init(forager, fit_names, fit_bounds, choice_history, reward_history, fit_method)
            fitting_parallel_results.append(result)
            
        # Find the global optimal
        cost = np.zeros(n_x0s)
        for nn,rr in enumerate(fitting_parallel_results):
            cost[nn] = rr.fun
        
        best_ind = np.argmin(cost)
        fitting_result = fitting_parallel_results[best_ind]
        
            
    # print(fitting_result, 'time = %g' % (time.time() - now))
    return fitting_result, fit_history






