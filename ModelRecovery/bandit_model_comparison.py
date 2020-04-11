# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:15:49 2020

@author: Han
"""

from bandit_model import BanditModel
from fitting_functions import fit_bandit

import numpy as np
import pandas as pd
import time

class BanditModelComparison:
    '''
    A new class that can define models, receive data, do fitting, and generate plots.
    This is the minimized module that can be plugged into Datajoint for real data.
    '''
    
    def __init__(self, data, models = None, 
                 fit_method = 'DE', fit_settings = {'DE_pop_size': 16}, pool = '', 
                 if_compute_predictive = False, if_compute_generative = False):
        
        if models is None:  
            # All models available. Use the format: {'forager':forager, 'para_name1': para_bounds1, 'para_name2': para_bounds2, ...}
            self.models = [
                            {'forager':'LossCounting', 
                              'loss_count_threshold_mean': [0,50],
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
            
        self.data = data
        self.choice_target = self.data['choice']
        self.reward_target = self.data['reward']
        self.K, self.n_trials = np.shape(self.reward_target)
        assert np.shape(self.choice_target)[1] == self.n_trials, 'Choice length should be equal to reward length!'
        
        self.if_compute_predictive = if_compute_predictive
        self.if_compute_generative = if_compute_generative
        self.fit_method = fit_method
        self.fit_settings = fit_settings
        self.pool = pool
        
        return
        
    def fit(self):
        
        self.fitting_results = []
        
        print('=== Model Comparison ===\nMethods = %s, %s, pool = %s' % (self.fit_method, self.fit_settings, self.pool!=''))
        for mm, model in enumerate(self.models):
            # == Get settings for this model ==
            forager = model['forager']
            fit_names = []
            fit_bounds = [[],[]]
            
            for pp in model:
                if pp != 'forager':
                    fit_names.append(pp) 
                    fit_bounds[0].append(model[pp][0])
                    fit_bounds[1].append(model[pp][1])
            
            # == Do fitting here ==
            print('Model %g/%g: %15s, Km = %g ...'%(mm+1, len(self.models), forager, np.sum(np.diff(np.array(fit_bounds),axis=0)>0)),end='')
            start = time.time()
                
            result_this = fit_bandit(forager, fit_names, fit_bounds, self.choice_target, self.reward_target, 
                                     self.fit_method, **self.fit_settings, pool = self.pool)[0]
            
            print(' AIC = %g, BIC = %g (done in %.3g secs)' % (result_this.AIC, result_this.BIC, time.time()-start) )
            self.fitting_results.append(result_this)
                
            # == Rerun a **PREDICTIVE** session ==
            if self.if_compute_predictive:
                #!!! To be finished
                bandit = BanditModel(**kwargs_all, fit_choice_history = choice_history, fit_reward_history = reward_history)
                bandit.simulate()

            # == Run a **GENERATIVE** session ==
            if self.if_compute_generative:
                #!!! This will need the actual schedule or random seed?
                pass 
            
            # == Calculate Bayes factor ==
            self.compute_Bayes_factor()
            
        return
    
    def compute_Bayes_factor(self):
        
        return
        
    def show_results(self, model):
                
        
        return
