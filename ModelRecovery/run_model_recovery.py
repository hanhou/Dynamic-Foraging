# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 17:15:59 2020

@author: Han
"""

import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar. HH

from models import BanditModels
from fitting_functions import fit_bandit, negLL_func
from plot_fitting import plot_para_recovery, plot_LL_surface
   
def fit_para_recovery(forager, para_names, para_bounds, n_models = 10, true_paras = None, n_trials = 1000, fit_method = 'DE', n_x0s = 1, pool = '', **kargs):
    n_paras = len(para_names)
    
    if true_paras is None:
        if_no_true_paras = True
        true_paras = np.zeros([n_paras, n_models])
    else:
        if_no_true_paras = False
        n_models = np.shape(true_paras)[1]
        
    fitted_paras = np.zeros([n_paras, n_models])

    # === Do para recovery ===        
    for n in tqdm(range(n_models), desc='Parameter Recovery, %s'%forager):
        # Generate simulated para using uniform distribution in para_bounds if not specified
        if if_no_true_paras: 
            true_paras_this = []
            for pp in range(n_paras):
                true_paras_this.append(np.random.uniform(para_bounds[0][pp], para_bounds[1][pp]))
            true_paras[:,n] = true_paras_this
            
        # Generate fake data
        choice_history, reward_history = generate_fake_data(forager, para_names, true_paras[:,n], **kargs)
            
        # Predictive fitting
        fitting_result = fit_bandit(forager, para_names, para_bounds, choice_history, reward_history, fit_method = fit_method, n_x0s = n_x0s, pool = pool)
        fitted_paras[:,n] = fitting_result.x
    
        # print(true_paras_this, fitting_result.x)
        
    # === Plot results ===
    plot_para_recovery(forager, true_paras, fitted_paras, para_names, para_bounds, n_trials, fit_method, n_x0s)
    
    return true_paras, fitted_paras

def generate_true_paras(para_bounds, n_models = 5, method = 'random_uniform'):
    
    if method == 'linspace':
        p1 = np.linspace(para_bounds[0][0], para_bounds[1][0], n_models[0])
        p2 = np.linspace(para_bounds[0][1], para_bounds[1][1], n_models[1])
        pp1,pp2 = np.meshgrid(p1,p2)
        true_paras = np.vstack([pp1.flatten(), pp2.flatten()])
        
        return true_paras

    elif method == 'random_uniform':
        n_paras = len(para_bounds[0])
        
        true_paras = np.zeros([n_paras, n_models])

        for n in range(n_models):
            true_paras_this = []
            for pp in range(n_paras):
                true_paras_this.append(np.random.uniform(para_bounds[0][pp], para_bounds[1][pp]))
            true_paras[:,n] = true_paras_this
                
        return true_paras

def generate_fake_data(forager, para_names, true_para, **kargs):
    # Generate fake data
    n_paras = len(para_names)
    karg_this = {}
    for pp in range(n_paras):
        karg_this[para_names[pp]] = true_para[pp]
    
    bandit = BanditModels(forager, **karg_this, **kargs)
    bandit.simulate()
    choice_history = bandit.choice_history
    reward_history = bandit.reward_history
    
    return choice_history, reward_history


def compute_LL_surface(forager, para_names, para_bounds, true_para, n_grid = [100,100], fit_method = 'DE', n_x0s = 1, pool = '', **kargs):
    '''
    Log-likelihood landscape (Fig.3a, Wilson 2019)

    '''
    n_try_paras = np.prod(n_grid)
    n_worker = int(mp.cpu_count())
    pool_surface = mp.Pool(processes = n_worker)
    
    p1 = np.linspace(para_bounds[0][0], para_bounds[1][0], n_grid[0])
    p2 = np.linspace(para_bounds[0][1], para_bounds[1][1], n_grid[1])

    # Make sure the true_paras are exactly on the grid
    true_para[0] = p1[np.argmin(np.abs(true_para[0] - p1))]
    true_para[1] = p2[np.argmin(np.abs(true_para[1] - p2))]
    
    # Generate fake data
    choice_history, reward_history = generate_fake_data(forager, para_names, true_para, **kargs)
    
    # Compute LL surface
    pp2, pp1 = np.meshgrid(p2,p1)  # Note the order
        
    LLs = np.zeros(n_try_paras)
    
    # Must use two separate for loops, one for assigning and one for harvesting!   
    pool_results = []
    for x,y in zip(np.nditer(pp1),np.nditer(pp2)):
        pool_results.append(pool_surface.apply_async(negLL_func, args = ([x, y], forager, para_names, choice_history, reward_history)))
        
    for nn,rr in tqdm(enumerate(pool_results), total = n_try_paras, desc='compute_LL_surface'):
        LLs[nn] = - rr.get()
        
    LLs = LLs.reshape(n_grid).T
    
    pool_surface.close()
    pool_surface.join()

    # Do fitting
    print('Fitting using %s, n_x0s = %g, pool = %s...'%(fit_method, n_x0s, pool!=''))
    fitting_result, fit_history = fit_bandit(forager, para_names, para_bounds, choice_history, reward_history, 
                                             fit_method = fit_method, n_x0s = n_x0s, pool = pool,
                                             if_history = True)
    
    # Plot LL surface and fitting history
    plot_LL_surface(LLs,fitting_result.x, true_para,fit_history,para_names,p1,p2, fit_method, n_x0s)
    
    return

#%%
if __name__ == '__main__':
    
    # Fitting methods: 
    # - Global optimizer (`DE`): use its own parallel method
    # - Local optimizer (`L-BFGS-B`, `SLSQP`, `TNC`, `trust-constr`): random initialization in parallel
    # - Speed: L-BFGS-B = SLSQP  >> TNC >>> trust-constr
    
    # --- Use async to run multiple initializations ---
   
    n_worker = int(mp.cpu_count()/2)
    pool = mp.Pool(processes = n_worker)
    
    n_trials = 1000
    
    forager = 'LossCounting'
    para_names = ['loss_count_threshold_mean','loss_count_threshold_std']
    para_bounds = [[0,0],[50,10]]
    
    # Para recovery
    # true_paras = generate_true_paras([[0,0],[30,5]], n_models = [5,5], method = 'linspace')
    
    # true_paras, fitted_para = fit_para_recovery(forager = forager, 
    #               para_names = para_names, para_bounds = para_bounds, 
    #               true_paras = true_paras, n_trials = n_trials, 
    #               fit_method = 'DE', n_x0s = 1, pool = pool);    
    
    n_trials = 1000

    forager = 'LossCounting'
    para_names = ['loss_count_threshold_mean','loss_count_threshold_std']
    para_bounds = [[0,0],[50,10]]
    
    # Para recovery
    true_paras = generate_true_paras([[0,0],[30,5]], n_models = [5,5], method = 'linspace')
    fit_para_recovery(forager = forager, 
                      para_names = para_names, para_bounds = para_bounds, 
                      true_paras = true_paras, n_trials = n_trials, 
                      fit_method = 'L-BFGS-B', n_x0s = 1, pool = '');    
    
    # LL_surface
    # compute_LL_surface(forager, para_names, para_bounds, n_grid = [20,20], true_para = [10,3], n_trials = n_trials, 
    #                     fit_method = 'DE', pool = pool)
    
    # compute_LL_surface(forager, para_names, para_bounds, n_grid = [20,20], true_para = [10,3], n_trials = n_trials, 
    #                     fit_method = 'L-BFGS-B', n_x0s = 8, pool = '')  # Show multiple histories from multiple initializations
    
    # compute_LL_surface(forager, para_names, para_bounds, n_grid = [20,20], true_para = [10,3], n_trials = n_trials, 
    #                     fit_method = 'L-BFGS-B', n_x0s = 8, pool = pool)

    pool.close()   # Just a good practice
    pool.join()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
