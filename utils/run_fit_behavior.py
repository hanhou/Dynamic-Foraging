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

def combine_each_model_comparison(objectA, objectB):
    '''
    Combine two BanditModelComparison objects

    '''
    import pandas as pd
    
    # Confirm they were from the same dataset
    assert(np.all(objectA.fit_choice_history == objectB.fit_choice_history))
    assert(np.all(objectA.fit_reward_history == objectB.fit_reward_history))

    # -- Add info in objectB into object A --
    objectA.models.extend(objectB.models) 
    objectA.results_raw.extend(objectB.results_raw) 
    
    new_pd = pd.concat([objectA.results, objectB.results], axis = 0)

    # -- Update table --
    delta_AIC = new_pd.AIC - np.min(new_pd.AIC) 
    delta_BIC = new_pd.BIC - np.min(new_pd.BIC)

    # Relative likelihood = Bayes factor = p_model/p_best = exp( - delta_AIC / 2)
    new_pd['relative_likelihood_AIC'] = np.exp( - delta_AIC / 2)
    new_pd['relative_likelihood_BIC'] = np.exp( - delta_BIC / 2)

    # Model weight = Relative likelihood / sum(Relative likelihood)
    new_pd['model_weight_AIC'] = new_pd['relative_likelihood_AIC'] / np.sum(new_pd['relative_likelihood_AIC'])
    new_pd['model_weight_BIC'] = new_pd['relative_likelihood_BIC'] / np.sum(new_pd['relative_likelihood_BIC'])
    
    # log_10 (Bayes factor) = log_10 (exp( - delta_AIC / 2)) = (-delta_AIC / 2) / log(10)
    new_pd['log10_BF_AIC'] = - delta_AIC/2 / np.log(10) # Calculate log10(Bayes factor) (relative likelihood)
    new_pd['log10_BF_BIC'] = - delta_BIC/2 / np.log(10) # Calculate log10(Bayes factor) (relative likelihood)
    
    new_pd['best_model_AIC'] = (new_pd.AIC == np.min(new_pd.AIC)).astype(int)
    new_pd['best_model_BIC'] = (new_pd.BIC == np.min(new_pd.BIC)).astype(int)
    
    new_pd.index = range(1,1+len(new_pd))
    
    # Update notations
    para_notation_with_best_fit = []
    for i, row in new_pd.iterrows():
        para_notation_with_best_fit.append('('+str(i)+') '+row.para_notation + '\n' + str(np.round(row.para_fitted,2)))

    new_pd['para_notation_with_best_fit'] = para_notation_with_best_fit

    objectA.results = new_pd
    objectA.results_sort = new_pd.sort_values(by='AIC')

    return objectA

def combine_group_results(raw_path = "..\\export\\", result_path = "..\\results\\model_comparison\\",
                          combine_prefix = ['model_comparison_', 'model_comparison_no_bias_'], save_prefix = 'model_comparison_15_'):
    '''
    Combine TWO runs of model comparison
    '''
    
    import pickle
   
    for r, _, f in os.walk(raw_path):
        for file in f:

            data_A = np.load(result_path + combine_prefix[0] + file, allow_pickle=True)
            data_A = data_A.f.results_each_mice.item()
            
            data_B = np.load(result_path + combine_prefix[1] + file, allow_pickle=True)
            data_B = data_B.f.results_each_mice.item()
            
            new_grand_mc = combine_each_model_comparison(data_A['model_comparison_grand'], data_B['model_comparison_grand'])
            
            new_session_wise_mc = []
            for AA, BB in zip(data_A['model_comparison_session_wise'], data_B['model_comparison_session_wise']):
                new_session_wise_mc.append(combine_each_model_comparison(AA, BB))
                
            # -- Save data --
            results_each_mice = {'model_comparison_grand': new_grand_mc, 'model_comparison_session_wise': new_session_wise_mc}
            np.savez_compressed( result_path + save_prefix + file, results_each_mice = results_each_mice)
            print('%s + %s: Combined!' %(combine_prefix[0] + file, combine_prefix[1] + file))
    

if __name__ == '__main__':
    
    n_worker = 8
    pool = mp.Pool(processes = n_worker)
    
    # ---
    # data = np.load("..\\export\\FOR01.npz")
    # model_comparison = fit_each_mice(data, pool = pool, models = [1,9], use_trials = np.r_[0:500])
    
    # --- Fit all mice, session-wise and pooling
    # fit_all_mice(path = '..\\export\\', save_prefix = 'model_comparison_no_bias' , models = [1,9,10,11,12,13,14,15], pool = pool)
    
    # --- Combine different runs ---
    # combine_group_results()
    
    
    
    pool.close()   # Just a good practice
    pool.join()
