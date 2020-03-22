# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:30:26 2020

@author: Han
"""
import numpy as np
import scipy.optimize as optimize
import multiprocessing as mp
from tqdm import tqdm  # For progress bar. HH

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

def callback_history(x, **kargs):
    '''
    Store the intermediate DE results. I have to use global variable as a workaround. Any better ideas?
    '''
    global fit_history
    fit_history.append(x)
    
    return

def fit_each_init(forager, fit_names, fit_bounds, choice_history, reward_history, fit_method, callback):
    x0 = []
    for lb,ub in zip(fit_bounds[0], fit_bounds[1]):
        x0.append(np.random.uniform(lb,ub))
        
    # Append the initial point
    callback_history(x0)
        
    fitting_result = optimize.minimize(negLL_func, x0, args = (forager, fit_names, choice_history, reward_history), method = fit_method,
                                       bounds = optimize.Bounds(fit_bounds[0], fit_bounds[1]), callback = callback, )
    return fitting_result


def fit_bandit(forager, fit_names, fit_bounds, choice_history, reward_history, if_history = False, fit_method = 'CG', n_x0s = 1, pool = ''):
    
    if if_history: 
        global fit_history
        fit_history = []
        fit_histories = []  # All histories for different initializations
    
    if fit_method == 'DE':
        
        # Use DE's own parallel method
        fitting_result = optimize.differential_evolution(func = negLL_func, args = (forager, fit_names, choice_history, reward_history),
                                                         bounds = optimize.Bounds(fit_bounds[0], fit_bounds[1]), 
                                                         mutation=(0.5, 1), recombination = 0.7, popsize = 16, strategy = 'best1bin', 
                                                         disp = False, 
                                                         workers = 1 if pool == '' else int(mp.cpu_count()/2),   # For DE, use pool to control if_parallel, although we don't use pool for DE
                                                         updating = 'immediate' if pool == '' else 'deferred',
                                                         callback = callback_history if if_history else None,)
        
        return fitting_result, [fit_history] if if_history else fitting_result
        
    elif fit_method in ['L-BFGS-B', 'SLSQP', 'TNC', 'trust-constr']:
        
        # Do parallel initialization
        fitting_parallel_results = []
        
        if pool != '':  # Go parallel
            pool_results = []
            
            # Must use two separate for loops, one for assigning and one for harvesting!
            for nn in range(n_x0s):
                # Assign jobs
                pool_results.append(pool.apply_async(fit_each_init, args = (forager, fit_names, fit_bounds, choice_history, reward_history, fit_method, 
                                                                            None)))   # We can have multiple histories only in serial mode
            for rr in pool_results:
                # Get data    
                fitting_parallel_results.append(rr.get())
        else:
            # Serial
            
            for nn in range(n_x0s):
                # We can have multiple histories only in serial mode
                if if_history: fit_history = []  # Clear this history
                
                result = fit_each_init(forager, fit_names, fit_bounds, choice_history, reward_history, fit_method, 
                                       callback = callback_history if if_history else None)
                
                fitting_parallel_results.append(result)
                if if_history: fit_histories.append(fit_history)
            
        # Find the global optimal
        cost = np.zeros(n_x0s)
        for nn,rr in enumerate(fitting_parallel_results):
            cost[nn] = rr.fun
        
        best_ind = np.argmin(cost)
        
        fitting_result = fitting_parallel_results[best_ind]
        if fit_histories != []:
            fit_histories.insert(0,fit_histories.pop(best_ind))  # Move the best one to the first
        
        return fitting_result, fit_histories if if_history else fitting_result
            






