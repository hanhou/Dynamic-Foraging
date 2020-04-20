# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 21:33:33 2020

@author: Han
"""
import numpy as np
import multiprocessing as mp
import os
import time
import sys
from tqdm import tqdm

from models.bandit_model_comparison import BanditModelComparison


def fit_each_mice(data, if_session_wise = False, if_verbose = True, file_name = '', pool = '', models = None, use_trials = None):
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
    
    n_trials = len(choice_history)
    print('Total valid trials = %g' % n_trials)
    sys.stdout.flush()
    
    reward_history = np.zeros([2,n_trials])
    for c in (0,1):  
        reward_history[c, choice_history == c] = (reward[choice_history == c] > 0).astype(int)
    
    choice_history = np.array([choice_history])
    
    results_each_mice = {}
    
    # -- Model comparison for each session --
    if if_session_wise:
        
        model_comparison_session_wise = []
        
        unique_session = np.unique(session_num)
        
        for ss in tqdm(unique_session, desc = 'Session-wise', total = len(unique_session)):
            choice_history_this = choice_history[:, session_num == ss]
            reward_history_this = reward_history[:, session_num == ss]

            if use_trials is not None:
                choice_history_this = choice_history_this[:, use_trials]
                reward_history_this = reward_history_this[:, use_trials]
                
            model_comparison_this = BanditModelComparison(choice_history_this, reward_history_this, models = models)
            model_comparison_this.fit(pool = pool, plot_predictive = None, if_verbose = False) # Plot predictive traces for the 1st, 2nd, and 3rd models
            model_comparison_session_wise.append(model_comparison_this)
                
        results_each_mice['model_comparison_session_wise'] = model_comparison_session_wise
    
    # -- Model comparison for all trials --
    # For debugging    
    if use_trials is not None:
        choice_history = choice_history[:, use_trials]
        reward_history = reward_history[:, use_trials]
        p_reward = p_reward[:,use_trials]
        session_num = session_num[use_trials]
    
    print('Pooling all sessions: ', end='')
    start = time.time()
    model_comparison_grand = BanditModelComparison(choice_history, reward_history, p_reward = p_reward, session_num = session_num, models = models)
    model_comparison_grand.fit(pool = pool, plot_predictive = None if if_session_wise else [1,2,3], if_verbose = if_verbose) # Plot predictive traces for the 1st, 2nd, and 3rd models
    print(' Done in %g secs' % (time.time() - start))
    
    if if_verbose:
        model_comparison_grand.show()
        model_comparison_grand.plot()
    
    results_each_mice['model_comparison_grand'] = model_comparison_grand    
    
    return results_each_mice

def fit_all_mice(path, save_prefix = 'model_comparison', pool = '', models = None):
    # -- Find all files --
    start_all = time.time()
    for r, _, f in os.walk(path):
        for file in f:
            data = np.load(os.path.join(r, file))
            print('=== Mice %s ===' % file)
            start = time.time()
            
            # Do it
            try:
                results_each_mice = fit_each_mice(data, file_name = file, pool = pool, models = models, if_session_wise = True, if_verbose = False)
                np.savez_compressed( path + save_prefix + '_%s' % file, results_each_mice = results_each_mice)
                print('Mice %s done in %g mins!\n' % (file, (time.time() - start)/60))
            except:
                print('SOMETHING WENT WRONG!!')
                
    print('\n ALL FINISHED IN %g hrs!' % ((time.time() - start_all)/3600) )
    

if __name__ == '__main__':
    
    n_worker = 8
    pool = mp.Pool(processes = n_worker)
    
    # ---
    # data = np.load("..\\export\\FOR01.npz")
    # model_comparison = fit_each_mice(data, pool = pool, models = [1,9], use_trials = np.r_[0:500])
    
    # --- Fit all mice, session-wise and pooling
    fit_all_mice(path = '..\\export\\', save_prefix = 'model_comparison_no_bias' , models = [1,9,10,11,12,13,14,15], pool = pool)
    
    pool.close()   # Just a good practice
    pool.join()
