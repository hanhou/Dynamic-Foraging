# =============================================================================
# Plotting functions for foraging_model_HH
#
# Feb 2020, Han Hou @ Houston
# Svoboda lab
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.gridspec import GridSpec

plt.rcParams.update({'font.size': 12})

LEFT = 0
RIGHT = 1

smooth_factor = 5

# matplotlib.use('Agg')  # Agg -> non-GUI backend. HH
# matplotlib.use('qt5agg')  # We can see the figure by qt5. HH

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_one_session(bandit, fig, plottype='2lickport'):
    
    # == Fetch data ==
    n_trials = bandit.n_trials
    choice_history = bandit.choice_history
    reward_history = bandit.reward_history
                                      
    rewarded_trials = np.any(reward_history, axis = 0)
    unrewarded_trials = np.logical_not(rewarded_trials)
    
    # == Choice trace ==
    if fig == '':
        fig = plt.figure()
        
    gs = GridSpec(2,3, top = 0.85)        
    ax = fig.add_subplot(gs[0,0:2])

    # Rewarded trials
    ax.plot(np.nonzero(rewarded_trials)[0], 0.5 + (choice_history[0,rewarded_trials]-0.5) * 1.4, 
            'k|',color='black',markersize=20, markeredgewidth=2)

    # Unrewarded trials
    ax.plot(np.nonzero(unrewarded_trials)[0], 0.5 + (choice_history[0,unrewarded_trials] - 0.5) * 1.4, 
            '|',color='gray', markersize=10, markeredgewidth=1)
    
    # Baited probability and smoothed choice history
    ax.plot(np.arange(0, n_trials), bandit.p_reward_fraction, color='DarkOrange', label = 'bait prob.')
    ax.plot(moving_average(choice_history, smooth_factor) , color='black', label = 'smoothed choice')
    
    # Q_estimation
    if bandit.forager not in ['Random', 'AlwaysLEFT', 'IdealGreedy', 'SuttonBartoRLBook']:
        ax.plot(moving_average(bandit.q_estimation[RIGHT,:], 1), color='Green', label = 'Q_estimation')
        
    ax.legend(fontsize = 10)
     
    ax.set_yticks([0,1])
    ax.set_yticklabels(['Left','Right'])
    
    # Efficiency
    plt.title('Example session, efficiency = %.3g%%' % (bandit.foraging_efficiency*100))
   
    # == Cumulative choice plot ==  [Sugrue 2004]
    bandit.cumulative_choice_L = np.cumsum(bandit.choice_history == LEFT)
    bandit.cumulative_choice_R = np.cumsum(bandit.choice_history == RIGHT)
    
    # Actual choices
    ax = fig.add_subplot(gs[0,2])
    ax.plot(bandit.cumulative_choice_L, bandit.cumulative_choice_R, color='black')
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    
    # p_rewards
    
    # bandit.block_trans_time = np.cumsum(np.hstack([0,bandit.block_size]))
    
    for i_block, block_start in enumerate(bandit.block_trans_time[:-1]):   # For each block in this session
        
        # Find the starting point and slope for each block
        
        x0 = bandit.cumulative_choice_L[block_start]
        y0 = bandit.cumulative_choice_R[block_start]
        slope = bandit.p_reward_ratio[block_start]    # Note that this should be p_reward_ratio, not p_reward_fraction!!
        
        # next_x = bandit.cumulative_choice_L[bandit.block_trans_time[i_block+1] - 1]   # To ensure horizontal continuity
        dx = bandit.block_size[i_block]/(1 + slope)   # To ensure total number of trials be the same
        dy = dx * slope
        
        # Plot p_reward_fraction
        ax.plot([x0 , x0 + dx], [y0, y0 + dy],'-', color='DarkOrange')
        
    plt.xlabel('Cumulative Left choices')
    plt.ylabel('Cumulative Right choices')
    plt.axis('square')

    return fig
    
def plot_all_reps(results_all_reps):
    
    fig = plt.figure(figsize=(12, 8))
        
    fig.text(0.05,0.94,'%s\n%g sessions, %g blocks, %g trials' % (results_all_reps['description'], 
                                                                results_all_reps['n_reps'], 
                                                                results_all_reps['n_blocks'], 
                                                                results_all_reps['n_trials']
                                                                ), fontsize = 15)
    fig.text(0.05,0.91,'Efficiency +/- 95%% CI: %.3g%% +/- %.2g%%' % (results_all_reps['foraging_efficiency'][0]*100,
                                                          results_all_reps['foraging_efficiency'][1]*100,
                                                          ), fontsize = 15)
    
    # == 1. Example Session ==
    if 'example_session' in results_all_reps:
        plot_one_session(results_all_reps['example_session'], fig)
    
    # == 2. Blockwise matching ==
    c_frac, r_frac, c_log_ratio, r_log_ratio = results_all_reps['blockwise_stats']
    
    gs = GridSpec(2,3, wspace=0.3, hspace=0.5, bottom=0.13)    
            
    if results_all_reps['example_session'].forager not in ['AlwaysLEFT','IdealGreedy'] \
        and not np.all(np.isnan(r_log_ratio)):
         
        # 2b. -- Log_ratio
        # ax = fig.add_subplot(235)
        ax = fig.add_subplot(gs[1,1])
        
        # Scatter plot
        ax.plot(r_log_ratio, c_log_ratio, '.k')

        # Get linear fit paras
        # "a,b" in Corrado 2005, "slope" in Iigaya 2019
        [a, a_CI95], [b, _],[r_square, p],[slope, slope_CI95] = results_all_reps['linear_fit_log_ratio'][0,:,:]

        # Plot line
        xx = np.linspace(min(r_log_ratio), max(r_log_ratio), 100)
        yy = np.log(b) + xx * a
        hh = ax.plot(xx,yy,'r')
        ax.legend(hh,['a = %.2g +/- %.2g\nr^2 = %.2g\np = %.2g' % (a, a_CI95, r_square, p)])
     
        plt.xlabel('Blockwise log reward ratio')
        plt.ylabel('Blockwise log choice ratio')
        # ax.set_aspect('equal','datalim')
        plt.axis('square')
    
        # 2a. -- Fraction
        # ax = fig.add_subplot(234)
        ax = fig.add_subplot(gs[1,0])
        ax.plot(r_frac, c_frac, '.k')
        ax.plot([0,1],[0,1],'k--')
        
        # Non-linear relationship using the linear fit of log_ratio
        xx = np.linspace(min(r_frac), max(r_frac), 100)
        yy = (xx ** a ) / (xx ** a + b * (1-xx) ** a)
        ax.plot(xx, yy, 'r')    
        
        # slope_fraction = 0.5 in theory
        yy = 1/(1+b) + (xx-0.5)*slope
        ax.plot(xx, yy, 'b--', linewidth=2, label='slope = %.3g' % slope)
        plt.legend()       
        
        plt.xlabel('Blockwise reward fraction')
        plt.ylabel('Blockwise choice fraction')
        plt.axis('square')
  
    # 2c. -- Stay duration distribution
    if np.sum(results_all_reps['stay_duration_hist']) > 0:
        ax = fig.add_subplot(gs[1,2])
        bin_center = np.arange(len(results_all_reps['stay_duration_hist']))
        ax.bar(bin_center + 0.5, results_all_reps['stay_duration_hist'] / np.sum(results_all_reps['stay_duration_hist']), 
               color = 'k', label = 'No COD')
        ax.set_yscale('log')
        plt.xlabel('Stay duration (trials)')
        plt.ylabel('Proportion')
        plt.legend()
    
    # fig.show()
    
  
def plot_para_scan(results_para_scan, para_to_scan, **kwargs):
    
    forager = results_para_scan['forager']
    n_reps = results_para_scan['n_reps']
    
    if len(para_to_scan) == 1:    # 1-D
        
        # === Reorganize data ==
        para_name, para_range = list(para_to_scan.items())[0]
        
        
        # Check para names in case it's taus, w_taus, etc...
        if isinstance(para_range,list):
            # Which one is change?
            para_diff = np.array(para_range[0]) - np.array(para_range[1])
            which_diff = np.where(para_diff)[0][-1]
            
            # Add workaround ...
            para_name = para_name + '_' + str(which_diff+1)
            para_range = np.array(para_range)[:, which_diff]

        
        para_diff = np.diff(para_range)
        if_log = para_diff[0] != para_diff[1]
        
        paras_foraging_efficiency = results_para_scan['foraging_efficiency_per_session']
        fe_mean = np.mean(paras_foraging_efficiency, axis = 1)
        fe_CI95 = 1.96 * np.std(paras_foraging_efficiency, axis = 1) / np.sqrt(n_reps)

        matching_slope = results_para_scan['linear_fit_log_ratio'][:,3,0]  # "Slope" in Iigaya 2019
        matching_slope_CI95 = results_para_scan['linear_fit_log_ratio'][:,3,1]
        
        # === Plotting ===
        gs = GridSpec(1,3, top = 0.85, wspace = 0.3, bottom = 0.12)
        fig = plt.figure(figsize=(12, 4))
        
        fig.text(0.05,0.94,'Forager = %s, n_repetitions = %g, %s' % (forager, n_reps, kwargs), fontsize = 15)
        
        # -- 1. Foraging efficiency vs para
        ax = fig.add_subplot(gs[0,0])
        plt.plot(para_range, fe_mean, '-')
        plt.fill_between(para_range, fe_mean - fe_CI95, fe_mean + fe_CI95, label = '95% CI', alpha = 0.2)
        plt.xlabel(para_name)
        plt.ylabel('Foraging efficiency')
        ax.legend()
        if if_log: ax.set_xscale('log')
        
        # Two baselines
        # Run with rep = 1000, separately 
        # Please change if you change the task structure !!!
        random_result = [0.772, 0.002]    
        ideal_result = [0.842, 0.0023]
        xlim = [para_range[0], para_range[-1]]
        
        plt.plot(xlim, [random_result[0]]*2, 'k--')
        plt.fill_between(xlim, - np.diff(random_result), np.sum(random_result), alpha = 0.2, color ='black')
        plt.text(xlim[0], random_result[0], 'random')
        
        plt.plot(xlim, [ideal_result[0]]*2, 'k--')
        plt.fill_between(xlim, - np.diff(ideal_result), np.sum(ideal_result), alpha = 0.2, color ='black')
        plt.text(xlim[0], ideal_result[0], 'ideal')

        # -- 2. Matching slope vs para
        ax = fig.add_subplot(gs[0,1])
        plt.plot(para_range, matching_slope, '-')
        plt.fill_between(para_range, matching_slope - matching_slope_CI95, matching_slope + matching_slope_CI95, label = '95% CI', alpha = 0.2)
        plt.xlabel(para_name)
        plt.ylabel('Matching slope (log ratio)')
        ax.legend()
        if if_log: ax.set_xscale('log')
        
        # -- 3. Foraging efficiency vs Matching slope
        ax = fig.add_subplot(gs[0,2])
        plt.plot(matching_slope, fe_mean, 'o-')
        plt.xlabel('Matching slope (log ratio)')
        plt.ylabel('Foraging efficiency')
        
    elif len(para_to_scan) == 2:    # 2-D
        
        # === Reorganize data ==
        para_names = list(para_to_scan.keys())
        para_ranges = list(para_to_scan.values())
        
        # Check para names in case it's taus, w_taus, etc...
        for nn, rr in enumerate(para_ranges):
            if isinstance(rr,list):
                # Which one is change?
                para_diff = np.array(rr[0]) - np.array(rr[1])
                which_diff = np.where(para_diff)[0][-1]
                
                # Add workaround ...
                para_names[nn] = para_names[nn] + '_' + str(which_diff+1)
                para_ranges[nn] = np.array(rr)[:, which_diff]
                
        
        # Reshape the results to 2-D
        paras_foraging_efficiency = results_para_scan['foraging_efficiency_per_session']
        fe_mean = np.mean(paras_foraging_efficiency, axis = 1).reshape(len(para_ranges[0]), len(para_ranges[1]))
        matching_slope = results_para_scan['linear_fit_log_ratio'][:,3,0].reshape(len(para_ranges[0]), len(para_ranges[1]))
        
        # === Plotting ===
        gs = GridSpec(1,3, top = 0.9, wspace = .6, bottom = 0.15)
        fig = plt.figure(figsize=(13, 4))
        
        fig.text(0.05,0.94,'Forager = %s, n_repetitions = %g, %s' % (forager, n_reps, kwargs), fontsize = 15)
        
        # -- 1. Foraging efficiency
        interp = 'none'
        cmap = plt.cm.get_cmap('hot')
        cmap.set_bad('cyan')
        
        x_label_idx = np.r_[0:len(para_ranges[0]):2]
        y_label_idx = np.r_[0:len(para_ranges[1]):2]
        
        ax = fig.add_subplot(gs[0,0])
        im = plt.imshow(fe_mean.T, interpolation=interp, cmap=cmap)
        
        plt.xlabel(para_names[0])
        ax.set_xlim(-0.5, len(para_ranges[0])-0.5)
        ax.set_xticks(x_label_idx)
        ax.set_xticklabels(np.round(para_ranges[0][x_label_idx],2))
        plt.xticks(rotation=45)
        
        plt.ylabel(para_names[1])
        ax.set_ylim(-0.5, len(para_ranges[1])-0.5)
        ax.set_yticks(y_label_idx)
        ax.set_yticklabels(np.round(para_ranges[1][y_label_idx],2))
        
        plt.title('Foraging efficiency')
        
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        # -- 2. Matching slope
        ax = fig.add_subplot(gs[0,1])
        im = plt.imshow(matching_slope.T, interpolation=interp, cmap=cmap)
        
        plt.xlabel(para_names[0])
        ax.set_xlim(-0.5, len(para_ranges[0])-0.5)
        ax.set_xticks(x_label_idx)
        ax.set_xticklabels(np.round(para_ranges[0][x_label_idx],2))
        plt.xticks(rotation=45)
        
        plt.ylabel(para_names[1])
        ax.set_ylim(-0.5, len(para_ranges[1])-0.5)
        ax.set_yticks(y_label_idx)
        ax.set_yticklabels(np.round(para_ranges[1][y_label_idx],2))
        
        plt.title('Matching slope (log ratio)')
        
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        # -- 3. Foraging efficiency vs Matching slope
        ax = fig.add_subplot(gs[0,2])
        plt.plot(matching_slope.T, fe_mean.T, 'o')
        
        plt.xlabel('Matching slope (log ratio)')
        plt.ylabel('Foraging efficiency')
               
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    