# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 17:15:59 2020

@author: Han
"""

import numpy as np
from tqdm import tqdm  # For progress bar. HH

from models import BanditModels
from fitting_functions import fit_bandit
from plot_fitting import plot_para_recovery
   
def fit_para_recovery(forager, para_names, para_bounds, n_models = 10, **kargs):
    simulated_para = np.zeros([len(para_names), n_models])
    fitted_para = np.zeros([len(para_names), n_models])

    # === Do para recovery ===        
    for n in tqdm(range(n_models), desc='Parameter Recovery, %s'%forager):
        # Generate simulated para
        simulated_para_this = []
        karg_this = {}
        for pp in range(len(para_names)):
            simulated_para_this.append(np.random.uniform(para_bounds[0][pp], para_bounds[1][pp]))
            karg_this[para_names[pp]] = simulated_para_this[pp]
            
        simulated_para[:,n] = simulated_para_this
        
        # Generate fake data
        bandit = BanditModels(forager, **karg_this, **kargs)
        bandit.simulate()
        choice_history = bandit.choice_history
        reward_history = bandit.reward_history
                
        # Predictive fitting
        fitting_result = fit_bandit(forager, para_names, para_bounds, choice_history, reward_history)
        fitted_para[:,n] = fitting_result.x
    
        # print(simulated_para_this, fitting_result.x)
        
    # === Plot results ===
              
    return simulated_para, fitted_para


#%%
    
if __name__ == '__main__':
    
    forager = 'LossCounting'
    para_names = ['loss_count_threshold_mean','loss_count_threshold_std']
    para_bounds = [[0,0],[50,10]]
    
    simulated_para, fitted_para = fit_para_recovery(forager = forager, 
                      para_names = para_names, para_bounds = para_bounds, 
                      n_models = 2, n_trials = 500)
    plot_para_recovery(forager, simulated_para, fitted_para, para_names, para_bounds)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
