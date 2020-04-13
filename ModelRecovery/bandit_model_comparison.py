# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:15:49 2020

@author: Han
"""

from fitting_functions import fit_bandit
from plot_fitting import plot_predictive_choice_prob

import numpy as np
import pandas as pd
import time

# All models available. Use the format: [forager, [para_names], [lower bounds], [higher bounds]]
MODELS = [
            ['LossCounting', ['loss_count_threshold_mean', 'loss_count_threshold_std'], [0,0], [40,10]],                   
            ['LNP_softmax',  ['tau1', 'softmax_temperature'], [1e-3, 1e-2], [100, 15]],                 
            ['LNP_softmax', ['tau1', 'tau2', 'w_tau1', 'softmax_temperature'],[1e-3, 1e-1, 0, 1e-2],[15, 40, 1, 15]],                 
            ['RW1972_epsi', ['learn_rate_rew', 'epsilon'],[0, 0],[1, 1]],
            ['RW1972_softmax', ['learn_rate_rew', 'softmax_temperature'],[0, 1e-2],[1, 15]],
            ['Bari2019', ['learn_rate_rew', 'forget_rate', 'softmax_temperature'],[0, 0, 1e-2],[1, 1, 15]],
            ['Hattori2019', ['learn_rate_rew', 'learn_rate_unrew', 'softmax_temperature'],[0, 0, 1e-2],[1, 1, 15]],
            ['Hattori2019', ['learn_rate_rew', 'learn_rate_unrew', 'forget_rate', 'softmax_temperature'],[0, 0, 0, 1e-2],[1, 1, 1, 15]],
         ]

# Define notations
PARA_NOTATIONS = {'loss_count_threshold_mean': '$\\mu_{LC}$',
            'loss_count_threshold_std': '$\\sigma_{LC}$',
            'tau1': '$\\tau_1$',
            'tau2': '$\\tau_2$',
            'w_tau1': '$w_{\\tau_1}$',
            'learn_rate_rew': '$\\alpha_{rew}$',   
            'learn_rate_unrew': '$\\alpha_{unr}$',   
            'forget_rate': '$\\delta$',
            'softmax_temperature': '$\\sigma$',
            'epsilon': '$\\epsilon$'
            }


class BanditModelComparison:
    '''
    A new class that can define models, receive data, do fitting, and generate plots.
    This is the minimized module that can be plugged into Datajoint for real data.
    '''
    
    def __init__(self, data, models = None):
    

        if models is None:  
            self.models = MODELS
        elif type(models[0]) is int:
            self.models = [MODELS[i-1] for i in models]
        else:
            self.models = models
            
        self.data = data # choice_history, reward_history, (schedule, if want to plot)
        self.fit_choice_history, self.fit_reward_history = self.data[0:2]
        self.K, self.n_trials = np.shape(self.fit_reward_history)
        assert np.shape(self.fit_choice_history)[1] == self.n_trials, 'Choice length should be equal to reward length!'
        
        return
        
    def fit(self, fit_method = 'DE', fit_settings = {'DE_pop_size': 16}, pool = '',
                  if_verbose = True, 
                  plot_predictive = None,  # E.g.: 0,1,2,-1: The best, 2nd, 3rd and the worst model
                  plot_generative = None):
        
        self.results_raw = []
        self.results = pd.DataFrame()
        
        if if_verbose: print('=== Model Comparison ===\nMethods = %s, %s, pool = %s' % (fit_method, fit_settings, pool!=''))
        for mm, model in enumerate(self.models):
            # == Get settings for this model ==
            forager, fit_names, fit_lb, fit_ub = model
            fit_bounds = [fit_lb, fit_ub]
            
            para_notation = ''
            Km = 0
            
            for name, lb, ub in zip(fit_names, fit_lb, fit_ub):
                # == Generate notation ==
                if lb < ub:
                    para_notation += PARA_NOTATIONS[name] + ', '
                    Km += 1
            
            para_notation = para_notation[:-2]
            
            # == Do fitting here ==
            #  Km = np.sum(np.diff(np.array(fit_bounds),axis=0)>0)
            
            if if_verbose: print('Model %g/%g: %15s, Km = %g ...'%(mm+1, len(self.models), forager, Km), end='')
            start = time.time()
                
            result_this = fit_bandit(forager, fit_names, fit_bounds, self.fit_choice_history, self.fit_reward_history, 
                                     fit_method = fit_method, **fit_settings, 
                                     pool = pool, if_predictive = plot_predictive is not None)
            
            if if_verbose: print(' AIC = %g, BIC = %g (done in %.3g secs)' % (result_this.AIC, result_this.BIC, time.time()-start) )
            self.results_raw.append(result_this)
            self.results = self.results.append(pd.DataFrame({'model': [forager], 'Km': Km, 'AIC': result_this.AIC, 'BIC': result_this.BIC, 
                                    'LPT_AIC': result_this.LPT_AIC, 'LPT_BIC': result_this.LPT_BIC,
                                    'para_names': [fit_names], 'para_bounds': [fit_bounds], 
                                    'para_notation': [para_notation], 'para_fitted': [np.round(result_this.x,3)]}, index = [mm+1]))
            
                
        # == Reorganize data ==
        self.results['log10_BF_AIC'] = (-(self.results.AIC - np.min(self.results.AIC))/2)/np.log(10) # Calculate Bayes factor 
        self.results['log10_BF_BIC'] = (-(self.results.BIC - np.min(self.results.BIC))/2)/np.log(10) # Calculate Bayes factor 
        self.results['best_model_AIC'] = (self.results.AIC == np.min(self.results.AIC)).astype(int)
        self.results['best_model_BIC'] = (self.results.BIC == np.min(self.results.BIC)).astype(int)
        self.results_sort = self.results.sort_values(by='AIC')
        
        # == Plotting == 
        if plot_predictive is not None: # Plot the predictive choice trace of the best fitting of the best model (Using AIC)
            self.plot_predictive = plot_predictive
            plot_predictive_choice_prob(self)

        return
    
    def show(self):
        pd.options.display.max_colwidth = 100
        display(self.results_sort[['model','Km', 'AIC','log10_BF_AIC', 'BIC','para_notation','para_fitted']].round(2))
        
