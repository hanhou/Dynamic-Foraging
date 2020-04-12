# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:15:49 2020

@author: Han
"""

from bandit_model import BanditModel
from fitting_functions import fit_bandit
from plot_fitting import plot_predictive_choice_prob

import numpy as np
import pandas as pd
import time

# Define notations
notation = {'loss_count_threshold_mean': '$\\mu_{LC}$',
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
    
    def __init__(self, data, models = None, 
                 fit_method = 'DE', fit_settings = {'DE_pop_size': 16}, pool = '', 
                 plot_predictive = None,  # E.g.: 0,1,2,-1: The best, 2nd, 3rd and the worst model
                 plot_generative = None):
        
        if models is None:  
            # All models available. Use the format: {'forager':forager, 'para_name1': para_bounds1, 'para_name2': para_bounds2, ...}
            self.models = [
                            {'forager':'LossCounting', 
                              'loss_count_threshold_mean': [0,40],
                              'loss_count_threshold_std': [0,10]},                   
                            {'forager':'LNP_softmax', 
                              'tau1': [1e-3,100],
                              'softmax_temperature': [1e-2,15]},                 
                            {'forager':'LNP_softmax', 
                              'tau1': [1e-3,15],
                              'tau2': [1e-1,40],
                              'w_tau1': [0,1],
                              'softmax_temperature': [1e-2,15]},                 
                            {'forager':'RW1972_epsi', 
                              'learn_rate_rew': [0,1],
                              'epsilon': [0,1]},
                            {'forager':'RW1972_softmax', 
                              'learn_rate_rew': [0,1],
                              'softmax_temperature': [1e-2,15]},
                            {'forager':'Bari2019', 
                              'learn_rate_rew': [0,1],
                              'forget_rate': [0,1],
                              'softmax_temperature': [1e-2,15]},
                            {'forager':'Hattori2019', 
                              'learn_rate_rew': [0,1],
                              'learn_rate_unrew': [0,1],
                              'softmax_temperature': [1e-2,15]},
                            {'forager':'Hattori2019', 
                              'learn_rate_rew': [0,1],
                              'learn_rate_unrew': [0,1],
                              'forget_rate': [0,1],
                              'softmax_temperature': [1e-2,15]},
                          ]

        else:
            self.models = models
            
        self.data = data # choice_history, reward_history, (schedule, if want to plot)
        self.fit_choice_history, self.fit_reward_history = self.data[0:2]
        self.K, self.n_trials = np.shape(self.fit_reward_history)
        assert np.shape(self.fit_choice_history)[1] == self.n_trials, 'Choice length should be equal to reward length!'
        
        self.plot_predictive = plot_predictive
        self.plot_generative = plot_generative
        self.fit_method = fit_method
        self.fit_settings = fit_settings
        self.pool = pool
        
        return
        
    def fit(self):
        
        self.results_raw = []
        self.results = pd.DataFrame()
        
        print('=== Model Comparison ===\nMethods = %s, %s, pool = %s' % (self.fit_method, self.fit_settings, self.pool!=''))
        for mm, model in enumerate(self.models):
            # == Get settings for this model ==
            forager = model['forager']
            fit_names = []
            fit_bounds = [[],[]]
            
            para_notation = ''
            Km = 0
            
            for pp in model:
                if pp != 'forager':
                    fit_names.append(pp) 
                    fit_bounds[0].append(model[pp][0])
                    fit_bounds[1].append(model[pp][1])
                    
                    # == Generate notation ==
                    if model[pp][0] < model[pp][1]:
                        para_notation += notation[pp] + ', '
                        Km += 1
            
            para_notation = para_notation[:-2]
            
            # == Do fitting here ==
            #  Km = np.sum(np.diff(np.array(fit_bounds),axis=0)>0)
            
            print('Model %g/%g: %15s, Km = %g ...'%(mm+1, len(self.models), forager, Km), end='')
            start = time.time()
                
            result_this = fit_bandit(forager, fit_names, fit_bounds, self.fit_choice_history, self.fit_reward_history, 
                                     fit_method = self.fit_method, **self.fit_settings, 
                                     pool = self.pool, if_predictive = self.plot_predictive is not None)
            
            print(' AIC = %g, BIC = %g (done in %.3g secs)' % (result_this.AIC, result_this.BIC, time.time()-start) )
            self.results_raw.append(result_this)
            self.results = self.results.append(pd.DataFrame({'model': [forager], 'Km': Km, 'AIC': result_this.AIC, 'BIC': result_this.BIC, 
                                    'LPT_AIC': result_this.LPT_AIC, 'LPT_BIC': result_this.LPT_BIC,
                                    'para_names': [fit_names], 'para_bounds': [fit_bounds], 
                                    'para_notation': [para_notation], 'para_fitted': [np.round(result_this.x,3)]}, index = [mm]))
            
                
        # == Reorganize data ==
        self.results['log10_BF_AIC'] = (-(self.results.AIC - np.min(self.results.AIC))/2)/np.log(10) # Calculate Bayes factor 
        self.results['log10_BF_BIC'] = (-(self.results.BIC - np.min(self.results.BIC))/2)/np.log(10) # Calculate Bayes factor 
        self.results['best_model_AIC'] = (self.results.AIC == np.min(self.results.AIC)).astype(int)
        self.results['best_model_BIC'] = (self.results.BIC == np.min(self.results.BIC)).astype(int)
        self.results.sort_values(by='AIC', inplace = True)
        
        # == Plotting == 
        if self.plot_predictive is not None: # Plot the predictive choice trace of the best fitting of the best model (Using AIC)
            plot_predictive_choice_prob(self)

        return
    
    def show(self):
        pd.options.display.max_colwidth = 100
        display(self.results[['model','Km', 'AIC','log10_BF_AIC', 'BIC','para_notation','para_fitted']].round(2))
        
