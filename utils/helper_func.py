# -*- coding: utf-8 -*-
"""
Created on Fri May  8 23:00:16 2020

@author: Han
"""
import numpy as np
import seaborn as sns
import matplotlib
from scipy.optimize import curve_fit


def softmax(x, softmax_temperature, bias = 0):
    
    # Put the bias outside /sigma to make it comparable across different softmax_temperatures.
    if len(x.shape) == 1:
        X = x/softmax_temperature + bias   # Backward compatibility
    else:
        X = np.sum(x/softmax_temperature, axis=0) + bias  # Allow more than one kernels (e.g., choice kernel)
    
    max_temp = np.max(X)
    
    if max_temp > 700: # To prevent explosion of EXP
        greedy = np.zeros(len(x))
        greedy[np.random.choice(np.where(X == np.max(X))[0])] = 1
        return greedy
    else:   # Normal softmax
        return np.exp(X)/np.sum(np.exp(X))  # Accept np.
    
def choose_ps(ps):
    '''
    "Poisson"-choice process
    '''
    ps = ps/np.sum(ps)
    return np.max(np.argwhere(np.hstack([-1e-16, np.cumsum(ps)]) < np.random.rand()))

def seaborn_style():
    """
    Set seaborn style for plotting figures
    """
    sns.set(style="ticks", context="paper", font_scale=1.4)
    # sns.set(style="ticks", context="talk", font_scale=2)
    sns.despine(trim=True)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
def moving_average(a, n=3) :
    ret = np.nancumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def fit_sigmoid_p_choice(p_reward, choice, win=10, stepsize=None):
    if stepsize is None: stepsize = win
    start_trial = 0
    mean_p_diff = []
    mean_choice_R_frac = []

    while start_trial + win <= len(choice):
        end_trial = start_trial + win
        
        mean_p_diff.append(np.mean(np.diff(p_reward[:, start_trial:end_trial], axis=0)))
        mean_choice_R_frac.append(np.sum(choice[start_trial:end_trial] == 1) / win)
        
        start_trial += stepsize
        
    mean_p_diff = np.array(mean_p_diff)
    mean_choice_R_frac = np.array(mean_choice_R_frac)

    p0 = [0, 1, 1, 0]

    popt, pcov = curve_fit(lambda x, x0, k: sigmoid(x, x0, k, a=1, b=0), 
                        mean_p_diff, 
                        mean_choice_R_frac, 
                        p0[:2], 
                        method='lm',
                        maxfev=10000)
    
    return popt, pcov, mean_p_diff, mean_choice_R_frac

def sigmoid(x, x0, k, a, b):
    y = a / (1 + np.exp(-k * (x - x0))) + b
    return y