# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 21:33:33 2020

@author: Han
"""
import numpy as np
import multiprocessing as mp

from models.bandit_model_comparison import BanditModelComparison


def behavior_model_comparison(data, use_trials = None, pool = '', models = None):
    choice = data.f.choice
    reward = data.f.reward
    p1 = data.f.p1
    p2 = data.f.p2
    session_num = data.f.session
    
    # -- Formating --
    # Remove ignores
    valid_trials = choice != 0
    
    choice_history = choice[valid_trials] - 1  # 1: LEFT, 2: RIGHT --> 0: LEFT, 1: RIGHT
    reward = reward[valid_trials]
    p_reward = np.vstack((p1[valid_trials],p2[valid_trials]))
    session_num = session_num[valid_trials]
    
    if use_trials is not None:
        choice_history = choice_history[use_trials]
        reward = reward[use_trials]
        p_reward = p_reward[:,use_trials]
        session_num = session_num[use_trials]
        
    n_trials = len(choice_history)
    print('valid trials = %g' % n_trials)
    
    reward_history = np.zeros([2,n_trials])
    for c in (0,1):  
        reward_history[c, choice_history == c] = (reward[choice_history == c] > 0).astype(int)
    
    choice_history = np.array([choice_history])
    
    # -- Model comparison --
    model_comparison = BanditModelComparison(choice_history, reward_history, p_reward = p_reward, session_num = session_num, models = models)
    model_comparison.fit(pool = pool, plot_predictive=[1,2,3]) # Plot predictive traces for the 1st, 2nd, and 3rd models
    model_comparison.show()
    model_comparison.plot()
    
    return model_comparison

if __name__ == '__main__':
    
    n_worker = 8
    pool = mp.Pool(processes = n_worker)
    
    data = np.load("..\\export\\FOR01.npz")
    model_comparison = behavior_model_comparison(data, pool = pool, models = [1,9], use_trials = np.r_[0:500])
    
    pool.close()   # Just a good practice
    pool.join()
