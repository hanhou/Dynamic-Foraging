# =============================================================================
# Plotting functions for foraging_model_HH
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

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
    plt.title('%s, efficiency = %.3g%%' % (bandit.description, bandit.foraging_efficiency*100), fontsize = 10)
    
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
    plt.axis('square')

    return fig
    
def plot_all_sessions(results_all_reps):
    
    fig = plt.figure(figsize=(12, 8))
    
    fig.text(0.05,0.97,'%g sessions, %g blks, %g trials' % (results_all_reps['n_sessions'], 
                                                            results_all_reps['n_blocks'], 
                                                            results_all_reps['n_trials']
                                                            ))
    fig.text(0.05,0.94,'Efficiency: %.3g%% +/- %.2g%%' % (results_all_reps['foraging_efficiency'][0]*100,
                                                          results_all_reps['foraging_efficiency'][1]*100,
                                                          ))
    
    # == 1. Example Session ==
    plot_one_session(results_all_reps['example_session'], fig)
    
    # == 2. Blockwise matching ==
    
    if not 'OCD' in results_all_reps['example_session'].forager:
        
        c_frac, r_frac, c_log_ratio, r_log_ratio = results_all_reps['blockwise_stats']
        
        # 2b. -- Log_ratio
        ax = fig.add_subplot(224)
        ax.plot(r_log_ratio, c_log_ratio, '.')
        
        x = r_log_ratio[~np.isnan(r_log_ratio)]
        y = c_log_ratio[~np.isnan(c_log_ratio)]
        
        # slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        model = sm.OLS(y, sm.add_constant(x)).fit()
        y_pred = model.predict()
        
        intercept, slope  = model.params
        intercept_CI95, slope_CI95  = np.diff(model.conf_int(), axis=1)/2
        r_square, p = (model.rsquared, model.pvalues)
        results_all_reps['linear_fit_log_ratio'] = np.block([[slope, slope_CI95], [intercept, intercept_CI95],[r_square, p[1]]])
        
        ax.plot(x,y_pred,'r')
        ax.text(0,min(plt.ylim()),'a = %.2g +/- %.2g\nr^2 = %.2g\np = %.2g' % (slope, slope_CI95, r_square, p[1]))
        
    
        plt.xlabel('Blockwise log reward ratio')
        plt.ylabel('Blockwise log choice ratio')
        plt.axis('square')
    
        
        # 2a. -- Fraction
        ax = fig.add_subplot(223)
        ax.plot(r_frac, c_frac, '.')
        ax.plot([0,1],[0,1],'k--')
        
        # Non-linear relationship using the linear fit of log_ratio
        a = slope
        b = np.exp(intercept)
        xx = np.linspace(min(r_frac), max(r_frac), 100)
        yy = (xx ** a ) / (xx ** a + b * (1-xx) ** a)
        ax.plot(xx, yy, 'r')    
        
        plt.xlabel('Blockwise reward fraction')
        plt.ylabel('Blockwise choice fraction')
        plt.axis('square')
        
  
    fig.show()
    
    return results_all_reps
  