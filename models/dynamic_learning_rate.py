# -*- coding: utf-8 -*-
"""
Created on Tue May 26 00:33:58 2020

@author: Han
"""

import numpy as np
import scipy.optimize as optimize
from tqdm import tqdm

from utils.helper_func import softmax


def fit_dynamic_learning_rate_session(choice_history, reward_history, slide_win = 10, pool = '', x0 = []):
    ''' Fit R-W 1972 with sliding window = 10 (Wang, ..., Botvinick, 2018) '''
    
    trial_n = np.shape(choice_history)[1]
    if x0 == []:    x0 = [0.4, 0.4, 0]
    
    # Settings for RW1972
    # ['RW1972_softmax', ['learn_rate', 'softmax_temperature', 'biasL'],[0, 1e-2, -5],[1, 15, 5]]
    fit_bounds = [[0, 1e-2, -5],[1, 15, 5]]
    
    Q = np.zeros(np.shape(reward_history))  # Cache of Q values (using the best fit at each step)
    choice_prob = Q.copy()
    fitted_learn_rate = np.zeros(np.shape(choice_history))
    fitted_sigma = np.zeros(np.shape(choice_history))
    fitted_bias = np.zeros(np.shape(choice_history))
    
    for t in tqdm(range(1, trial_n - slide_win), desc = 'Sliding window', total = trial_n - slide_win):
    # for t in range(1, trial_n - slide_win):  # Start from the second trial
        Q_0 = Q[:, t-1] # Initial Q for this window
        choice_this = choice_history[:, t : t + slide_win]
        reward_this = reward_history[:, t : t + slide_win]
        
        fitting_result = optimize.differential_evolution(func = negLL_slide_win, args = (Q_0, choice_this, reward_this),
                                                  bounds = optimize.Bounds(fit_bounds[0], fit_bounds[1]), 
                                                  mutation=(0.5, 1), recombination = 0.7, popsize = 8, strategy = 'best1bin', 
                                                  disp = False, 
                                                  workers = 1 if pool == '' else 8,   # For DE, use pool to control if_parallel, although we don't use pool for DE
                                                  updating = 'immediate' if pool == '' else 'deferred')
        
        # fitting_result = optimize.minimize(negLL_slide_win, x0, args = (Q_0, choice_this, reward_this), method = 'L-BFGS-B', 
        #                   bounds = optimize.Bounds(fit_bounds[0], fit_bounds[1]))

        # Save parameters
        learn_rate, softmax_temperature, biasL = fitting_result.x
        fitted_learn_rate[:, t] = learn_rate
        fitted_sigma[:, t] = softmax_temperature
        fitted_bias[:, t] = biasL
        
        # Simulate one step to get the first Q from this best fit as the initial value of the next window
        choice_0 = choice_this[0, 0]
        Q[choice_0, t] = Q_0[choice_0] + learn_rate * (reward_this[choice_0, 0] - Q_0[choice_0])  # Chosen side
        Q[1 - choice_0, t] = Q_0[1 - choice_0]   # Unchosen side
        
        choice_prob[:, t] = softmax(Q[:, t], softmax_temperature, bias = np.array([biasL, 0]))    # Choice prob (just for validation)
        
    return fitted_learn_rate, fitted_sigma, fitted_bias, Q, choice_prob
    
        
def negLL_slide_win(fit_value, *args):
    '''    Negative likelihood function for the sliding window    '''
    
    # Arguments interpretation
    Q_0, choices, rewards = args
    learn_rate, softmax_temperature, biasL = fit_value
    bias_terms = np.array([biasL, 0])
    
    trial_n_win = np.shape(choices)[1]
    Q_win = np.zeros_like(rewards)  # K_arm * trial_n
    choice_prob_win = np.zeros_like(rewards)
    
    # -- Do mini-simulation in this sliding window (light version of RW1972) --
    for t in range(trial_n_win):
        Q_old = Q_0 if t == 0 else Q_win[:, t - 1]
        
        # Update Q
        choice_this = choices[0, t]
        Q_win[choice_this, t] = Q_old[choice_this] + learn_rate * (rewards[choice_this, t] - Q_old[choice_this])  # Chosen side
        Q_win[1 - choice_this, t] = Q_old[1 - choice_this]   # Unchosen side
        
        # Update choice_prob
        choice_prob_win[:, t] = softmax(Q_win[:, t], softmax_temperature, bias = bias_terms)  
        
    # Compute negative likelihood
    likelihood_each_trial = choice_prob_win [choices[0,:], range(trial_n_win)]  # Get the actual likelihood for each trial
    
    # Deal with numerical precision
    likelihood_each_trial[(likelihood_each_trial <= 0) & (likelihood_each_trial > -1e-5)] = 1e-16  # To avoid infinity, which makes the number of zero likelihoods informative!
    likelihood_each_trial[likelihood_each_trial > 1] = 1
    
    negLL = - sum(np.log(likelihood_each_trial))
        
    return negLL
