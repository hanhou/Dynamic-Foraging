# =============================================================================
# Plotting functions for foraging_model_HH
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({'font.size': 13})

LEFT = 0
RIGHT = 1

# matplotlib.use('Agg')  # Agg -> non-GUI backend. HH
# matplotlib.use('qt5agg')  # We can see the figure by qt5. HH

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_one_session(bandit, fig, plottype='2lickport'):     # Part of code from Marton
    
    # == Fetch data ==
    n_trials = bandit.n_trials
    choice_history = bandit.choice_history
    reward_history = bandit.reward_history
                                      
    rewarded_trials = np.any(reward_history, axis = 0)
    unrewarded_trials = np.logical_not(rewarded_trials)
    
    # == Choice trace ==
    if fig == '':
        fig = plt.figure()
        
    gs = GridSpec(3,3)        
    ax = fig.add_subplot(gs[0,0:2])

    # Rewarded trials
    ax.plot(np.nonzero(rewarded_trials)[0], 0.5 + (choice_history[rewarded_trials]-0.5) * 1.4, 
            'k|',color='black',markersize=20, markeredgewidth=2)
    
    # Unrewarded trials
    ax.plot(np.nonzero(unrewarded_trials)[0], 0.5 + (choice_history[unrewarded_trials] - 0.5) * 1.4, 
            '|',color='gray', markersize=10, markeredgewidth=1)
    
    # Baited probability and smoothed choice history
    ax.plot(np.arange(0, n_trials), bandit.p_reward_fraction, color='DarkOrange')
    ax.plot(moving_average(choice_history, 10) , color='black')
    
    ax.set_yticks([0,1])
    ax.set_yticklabels(['Left','Right'])
    plt.xlabel('Example session')
    
    # Reward rate
    plt.title('%s, efficiency = %.02f' % (bandit.description, bandit.foraging_efficiency), fontsize = 10)
    
    # == Cumulative choice plot ==  [Sugrue 2004]
    bandit.cumulative_choice_L = np.cumsum(bandit.choice_history == LEFT)
    bandit.cumulative_choice_R = np.cumsum(bandit.choice_history == RIGHT)
    
    # Actual choices
    ax = fig.add_subplot(gs[0,2])
    ax.plot(bandit.cumulative_choice_L, bandit.cumulative_choice_R, color='black')
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    
    # p_rewards
    
    # bandit.block_trans_time = np.cumsum(np.hstack([0,bandit.n_trials_per_block]))
    
    for i_block, block_start in enumerate(bandit.block_trans_time[:-1]):   # For each block in this session
        
        # Find the starting point and slope for each block
        
        x0 = bandit.cumulative_choice_L[block_start]
        y0 = bandit.cumulative_choice_R[block_start]
        slope = bandit.p_reward_ratio[block_start]    # Note that this should be p_reward_ratio, not p_reward_fraction!!
        
        # next_x = bandit.cumulative_choice_L[bandit.block_trans_time[i_block+1] - 1]   # To ensure horizontal continuity
        dx = bandit.n_trials_per_block[i_block]/(1 + slope)   # To ensure total number of trials be the same
        dy = dx * slope
        
        # Plot p_reward_fraction
        ax.plot([x0 , x0 + dx], [y0, y0 + dy],'-', color='DarkOrange')
        
    plt.xlabel('Cumulative Left choices')
    plt.ylabel('Cumulative Right choices')
    ax.set_aspect('equal')

    return fig
    
def plot_all_sessions(results_all_sessions):
    
    fig = plt.figure(figsize=(12, 8))
    
    # == 1. Example Session ==
    plot_one_session(results_all_sessions[-1], fig)   # Plot the last example session
    
    # == 2. Blockwise matching ==
    
    # choice_all_runs = np.zeros([n_sessions, bandit.n_trials])
    # reward_all_runs = np.zeros([n_sessions, global_k_arm, bandit.n_trials])  # Assuming all bandits have the same n_trials
 
     
    
    # 2a.  
    
    fig.show()
    return
  