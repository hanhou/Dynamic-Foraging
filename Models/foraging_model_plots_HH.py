# =============================================================================
# Plotting functions for foraging_model_HH
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})

LEFT = 0
RIGHT = 1

# matplotlib.use('Agg')  # Agg -> non-GUI backend. HH
# matplotlib.use('qt5agg')  # We can see the figure by qt5. HH

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_one_session(results_this_run, plottype='2lickport'):     # Part of code from Marton
    
    # Fetch data
    n_trials = results_this_run['n_trials']
    p_reward = results_this_run['p_reward']
    reward_available = results_this_run['reward_available']
    choice_history = results_this_run['choice_history']
    reward_history = results_this_run['reward_history']
    description = results_this_run['description']
                                      
    rewarded_trials = np.any(reward_history, axis = 0)
    unrewarded_trials = np.logical_not(rewarded_trials)
    
    # Foraging efficiency = Sum of actual rewards / Maximum number of rewards that could have been collected
    actual_reward_rate = np.sum(reward_history) / n_trials
    
    '''Don't know which one is better'''
    # maximum_reward_rate = np.mean(np.max(p_reward, axis = 0)) #??? Method 1: Average of max(p_reward) 
    maximum_reward_rate = np.mean(np.sum(p_reward, axis = 0)) #??? Method 2: Average of sum(p_reward).   [Corrado et al 2005: efficienty = 50% for choosing only one color]
    # maximum_reward_rate = np.sum(np.any(reward_available, axis = 0)) / n_trials  #??? Method 3: Maximum reward given the fixed reward_available (one choice per trial constraint) [Sugrue 2004???]
    # maximum_reward_rate = np.sum(np.sum(reward_available, axis = 0)) / n_trials  #??? Method 4: Sum of all ever-baited rewards (not fair)  

    foraging_efficiency = actual_reward_rate / maximum_reward_rate
    
    p_reward_ratio = p_reward[RIGHT,:] / (np.sum(p_reward, axis = 0))
    
    fig = plt.figure()
    ax1 = plt.subplot(2,1,1)

    # Rewarded trials
    ax1.plot(np.nonzero(rewarded_trials)[0], choice_history[rewarded_trials], 'k|',color='black',markersize=30, markeredgewidth=2)
    
    # Unrewarded trials
    ax1.plot(np.nonzero(unrewarded_trials)[0], choice_history[unrewarded_trials], '|',color='gray', markersize=15, markeredgewidth=1)
    
    # Baited probability and smoothed choice history
    ax1.plot(np.arange(0, n_trials), p_reward_ratio, color='orange')
    ax1.plot(moving_average(choice_history, 10) , color='black')
    
    ax1.set_yticks([0,1])
    ax1.set_yticklabels(['Left','Right'])
    plt.xlabel('choice #')
    
    # Reward rate
    plt.title('%s, efficiency = %.02f' % (description, foraging_efficiency), fontsize = 10)
    fig.show()
    
def plot_all_sessions():
    return
  