# =============================================================================
# Plotting functions for foraging_model_HH
#
# Feb 2020, Han Hou @ Houston
# Svoboda lab
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import plotly.express as px
from scipy import ndimage

from matplotlib.gridspec import GridSpec
from utils.helper_func import seaborn_style, sigmoid   
from utils.descriptive_analysis import plot_logistic_regression, plot_wsls, decode_betas


plt.rcParams.update({'font.size': 13})

LEFT = 0
RIGHT = 1

# matplotlib.use('Agg')  # Agg -> non-GUI backend. HH
# matplotlib.use('qt5agg')  # We can see the figure by qt5. HH

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def get_baseline(if_baited, p_reward_sum, p_reward_pairs):
    
    if if_baited and p_reward_sum == 0.45 and p_reward_pairs == None:
        random_result = [0.772, 0.002]    
        ideal_result = [0.842, 0.0023]
    elif not if_baited and p_reward_sum == 0.45 and p_reward_pairs == None:
        random_result = [0.499, 0.19]    
        ideal_result = [0.767, 0.003]
    elif if_baited and p_reward_sum == 0.8 and p_reward_pairs == None:
        random_result = [0.661, 0.001]    
        ideal_result = [0.813, 0.002]
    else:  # p_reward_pairs == [0.45, 0]
        random_result = [0.694, 0.002]    
        ideal_result = [0.999, 0.002]
        
        
    return random_result, ideal_result
        

def plot_one_session(bandit, fig='', plottype='2lickport'):
    
    if bandit.forager in ['IdealpHatOptimal', 'IdealpHatGreedy', 'FullStateQ'] or 'PatternMelioration' in bandit.forager:
        smooth_factor = 1
    else:
        # smooth_factor = 1
        smooth_factor = 5
    
    # == Fetch data ==
    n_trials = bandit.n_trials
    choice_history = bandit.choice_history
    reward_history = bandit.reward_history
    
    # == Choice trace ==
    if fig == '':
        fig = plt.figure()
        
    # gs = GridSpec(2,3, top = 0.85)        
    # ax = fig.add_subplot(gs[0,0:2])

    gs = GridSpec(2, 3, top = 0.80, width_ratios=[1, 1, 1.5])        
    ax = fig.add_subplot(gs[0, :])

    if not bandit.if_varying_amplitude:                                    
        rewarded_trials = np.any(reward_history, axis = 0)
        unrewarded_trials = np.logical_not(rewarded_trials)
        
        # Rewarded trials
        ax.plot(np.nonzero(rewarded_trials)[0], 0.5 + (choice_history[0,rewarded_trials]-0.5) * 1.4, 
                '|',color='black',markersize=20, markeredgewidth=2)
        # Unrewarded trials
        ax.plot(np.nonzero(unrewarded_trials)[0], 0.5 + (choice_history[0,unrewarded_trials] - 0.5) * 1.4, 
                '|',color='gray', markersize=10, markeredgewidth=1)
    else:  # Varying amplitude
        ax.scatter(np.arange(0, n_trials), 0.5 + (choice_history[0,:]-0.5) * 1.4, s = 500 * np.sum(reward_history, axis=0),
                marker = '|', color='black')        
        
    
    # Base probability
    ax.plot(np.arange(0, len(bandit.p_reward_fraction)), bandit.p_reward_fraction, color='DarkOrange', label = 'base rew. prob.')
    
    # Choice probability
    if bandit.forager in ['LossCounting']:
        ax.plot(bandit.loss_count[0,:] / 10, color='Blue', label = 'loss count')

    elif 'PatternMelioration' in bandit.forager:
        ax.plot(moving_average(bandit.q_estimation[RIGHT,:], 1), 'g-', label = 'Q_R')
        ax.plot(moving_average(bandit.q_estimation[LEFT,:], 1), 'g--', label = 'Q_L')
        
    elif bandit.forager not in ['Random', 'IdealpHatOptimal', 'IdealpHatGreedy', 'pMatching', 'AlwaysLEFT', 
                                'IdealpGreedy', 'SuttonBartoRLBook','AmB1'] and 'FullState' not in bandit.forager:
        ax.plot(moving_average(bandit.choice_prob[RIGHT,:], 1), color='Green', label = 'choice prob.')

    # Smoothed choice history
    ax.plot(moving_average(choice_history, smooth_factor) , color='black', label = 'choice (smooth = %g)' % smooth_factor)
        
    ax.legend(fontsize = 10)
     
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Left','Right'])
    
    # Efficiency
    plt.title('Example session, efficiency = %.3g%%' % (bandit.foraging_efficiency*100))
    
    # == Psychometric curve ==
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(bandit.psychometric_mean_p_diff, 
             bandit.psychometric_mean_choice_R_frac, 
             'ok', alpha=0.2)
    
    xx = np.linspace(-1, 1, 100)
    yy = sigmoid(xx, *bandit.psychometric_popt, a=1, b=0)
    ax1.plot(xx, yy, 'k--')
    ax1.set(xlabel=f'$p_R - p_L$ (win = {bandit.psychometric_win} trials)', ylabel='Fraction choose right')
    
    # == WSLS ===
    ax3 = fig.add_subplot(gs[1, 1])
    # plot_wsls(bandit.p_wsls, ax=ax3)
    
    # == Logistic regression ==
    if hasattr(bandit, 'logistic_reg'):
        ax2 = fig.add_subplot(gs[1, 2])
        plot_logistic_regression(bandit.logistic_reg, ax=ax2, ls='--')
        ax2.set(title='')
    
    # == Cumulative choice plot ==  [Sugrue 2004]
    # bandit.cumulative_choice_L = np.cumsum(bandit.choice_history == LEFT)
    # bandit.cumulative_choice_R = np.cumsum(bandit.choice_history == RIGHT)
    
    # # Actual choices
    # ax = fig.add_subplot(gs[0,2])
    # ax.plot(bandit.cumulative_choice_L, bandit.cumulative_choice_R, color='black')
    # ax.yaxis.set_label_position('right')
    # ax.yaxis.tick_right()
    
    # # p_rewards
    
    # # bandit.block_trans_time = np.cumsum(np.hstack([0,bandit.block_size]))
    
    # for i_block, block_start in enumerate(bandit.block_trans_time[:-1]):   # For each block in this session
        
    #     # Find the starting point and slope for each block
        
    #     x0 = bandit.cumulative_choice_L[block_start]
    #     y0 = bandit.cumulative_choice_R[block_start]
    #     slope = bandit.p_reward_ratio[block_start]    # Note that this should be p_reward_ratio, not p_reward_fraction!!
        
    #     # next_x = bandit.cumulative_choice_L[bandit.block_trans_time[i_block+1] - 1]   # To ensure horizontal continuity
    #     dx = bandit.block_size[i_block]/(1 + slope)   # To ensure total number of trials be the same
    #     dy = dx * slope
        
    #     # Plot p_reward_fraction
    #     ax.plot([x0 , x0 + dx], [y0, y0 + dy],'-', color='DarkOrange')
        
    # plt.xlabel('Cumulative Left choices')
    # plt.ylabel('Cumulative Right choices')
    # plt.axis('square')

    return fig
    
def plot_all_reps(results_all_reps):
    
    if_restless = 'restless' in results_all_reps['bandits_all_sessions'][0].task
    
    fig = plt.figure(figsize=(15*1, 4.5*2))
        
    if if_restless:
        fig.text(0.05,0.94,'%s\n%g sessions, %g blocks, %g trials, restless sigma = %g' % (results_all_reps['description'], 
                                                                    results_all_reps['n_reps'], 
                                                                    results_all_reps['n_blocks'], 
                                                                    results_all_reps['n_trials'], 
                                                                    results_all_reps['bandits_all_sessions'][0].sigma                                                                
                                                                    ), fontsize = 15)
    else:
        fig.text(0.05,0.94,'%s\n%g sessions, %g blocks, %g trials, p_override = %s' % (results_all_reps['description'], 
                                                                    results_all_reps['n_reps'], 
                                                                    results_all_reps['n_blocks'], 
                                                                    results_all_reps['n_trials'], 
                                                                    results_all_reps['p_reward_pairs'],
                                                                    ), fontsize = 15)
    
    if results_all_reps['if_baited']:
        if results_all_reps['if_varying_amplitude']:
            baiting_method = 'amp.'
        else:
            baiting_method = 'prob.'
    else:
        baiting_method = ''
        
    if if_restless:
        fig.text(0.05,0.88,'Efficiency +/- 95%% CI: %.3g%% +/- %.2g%% (if_baited = %s (%s))' % (results_all_reps['foraging_efficiency'][0]*100,
                                                        results_all_reps['foraging_efficiency'][1]*100, results_all_reps['if_baited'], baiting_method, 
                                                        ), fontsize = 15)
    else:
        fig.text(0.05,0.88,'Efficiency +/- 95%% CI: %.3g%% +/- %.2g%% (if_baited = %s (%s), p_reward_sum = %g)' % (results_all_reps['foraging_efficiency'][0]*100,
                                                            results_all_reps['foraging_efficiency'][1]*100, results_all_reps['if_baited'], baiting_method, 
                                                            results_all_reps['p_reward_sum']
                                                            ), fontsize = 15)
    
    # == 1. Example Session ==
    if 'example_session' in results_all_reps:
        plot_one_session(results_all_reps['example_session'], fig)
    
    # == 2. Plot all psychometric curves ==
    ax1 = fig.get_axes()[1]
    xx = np.linspace(-1, 1, 100)
    yy = []
    
    for bandit in results_all_reps['bandits_all_sessions']:
        yy.append(sigmoid(xx, *bandit.psychometric_popt, a=1, b=0))
        ax1.plot(xx, yy[-1], 'k-', alpha=0.1)  # Fitting from other sessions
    
    ax1.plot(xx, np.mean(yy, axis=0), 'k', lw=4)   # Average fitting
    
    # == 3. WSLS ===
    ax2 = fig.get_axes()[2]
    p_lookup = ('p_stay_win',
            'p_stay_win_L',
            'p_stay_win_R',
            'p_switch_lose',
            'p_switch_lose_L',
            'p_switch_lose_R',
            )
    
    # Fake group data to reuse the plotting function 
    p_wsls = results_all_reps['bandits_all_sessions'][0].p_wsls
    
    for name in p_wsls.keys():
        if '_CI' in name: continue
        
        cache = []
        for bandit in results_all_reps['bandits_all_sessions']:
            cache.append(bandit.p_wsls[name])
        
        p_wsls[name] = np.mean(cache)
        p_wsls[name + '_CI'] = np.std(cache, axis=0)/np.sqrt(results_all_reps['n_reps'])
    
    plot_wsls(p_wsls, ax=ax2)
    
    # == 4. Plot all logistic regressions ==
    if_logistic = hasattr(results_all_reps['bandits_all_sessions'][0], 'logistic_reg')
    
    if if_logistic:
        ax2 = fig.get_axes()[3]
        
        # Fake a CV class using fittings across simulation runs 
        logistic_reg = results_all_reps['bandits_all_sessions'][0].logistic_reg
        
        cache = {}
        betas = ['b_RewC', 'b_UnrC', 'b_C', 'bias']
        
        for beta in betas:
            cache = []
            for bandit in results_all_reps['bandits_all_sessions']:
                cache.append(getattr(bandit.logistic_reg, beta)[0, :])
            
            mean = np.mean(cache, axis=0)
            sem = np.std(cache, axis=0) / np.sqrt(results_all_reps['n_reps']) * 1.96
            setattr(logistic_reg, beta, np.atleast_2d(mean))
            setattr(logistic_reg, beta + '_CI', 
                    np.stack([mean - sem, mean + sem]))
            
        #logistic_reg.scores_ = {1.0: [bandit.logistic_reg.test_score for bandit in results_all_reps['bandits_all_sessions']]}
            
        plot_logistic_regression(logistic_reg, ax=ax2)
        
        h, l = ax2.get_legend_handles_labels()
        ax2.legend(h[-4:], [Rf'$\beta$_{a} $\pm$ sem' for a in ('RewC', 'UnrC', 'C', 'bias')])
        ax2.set(title='')
            
    
    # # == 2. Blockwise matching ==
    # c_frac, inc_frac, c_log_ratio, inc_log_ratio, rtn_log_ratio = results_all_reps['blockwise_stats']
    
    # gs = GridSpec(2,3, wspace=0.3, hspace=0.5, bottom=0.13)    
            
    # # if not np.all(np.isnan(inc_log_ratio)):
         
    # # 2b. -- Log_ratio
    # # ax = fig.add_subplot(235)
    # ax = fig.add_subplot(gs[1,1])
    
    # # Scatter plot
    # ax.plot(inc_log_ratio, c_log_ratio, '.k')

    # # Get linear fit paras
    # # "a,b" in Corrado 2005, "slope" in Iigaya 2019
    # [a, a_CI95], [b, _],[r_square, p],[slope, slope_CI95] = results_all_reps['linear_fit_log_income_ratio'][0,:,:]

    # # Plot line
    # xx = np.linspace(min(inc_log_ratio), max(inc_log_ratio), 100)
    # yy = np.log(b) + xx * a
    # hh = ax.plot(xx,yy,'r')
    # ax.legend(hh,['a = %.2g +/- %.2g\nr^2 = %.2g\np = %.2g' % (a, a_CI95, r_square, p)])
    
    # plt.plot([-4,4],[-4,4],'y--')
 
    # plt.xlabel('Blockwise log income ratio')
    # plt.ylabel('Blockwise log choice ratio')
    # # ax.set_aspect('equal','datalim')
    # plt.axis('square')

    # # 2a. -- Fraction
    # # ax = fig.add_subplot(234)
    # ax = fig.add_subplot(gs[1,0])
    # ax.plot(inc_frac, c_frac, '.k')
    # ax.plot([0,1],[0,1],'y--')
    
    # # Non-linear relationship using the linear fit of log_ratio
    # xx = np.linspace(min(inc_frac), max(inc_frac), 100)
    # yy = (xx ** a ) / (xx ** a + b * (1-xx) ** a)
    # ax.plot(xx, yy, 'r')    
    
    # # slope_fraction = 0.5 in theory
    # yy = 1/(1+b) + (xx-0.5)*slope
    # ax.plot(xx, yy, 'b--', linewidth=2, label='slope = %.3g' % slope)
    # plt.legend()       
    
    # plt.xlabel('Blockwise INCOME fraction')
    # plt.ylabel('Blockwise choice fraction')
    # plt.axis('square')
  
    # 2c. -- Stay duration distribution
    # if np.sum(results_all_reps['stay_duration_hist']) > 0:
    #     ax = fig.add_subplot(gs[1,2])
    #     bin_center = np.arange(len(results_all_reps['stay_duration_hist']))
    #     ax.bar(bin_center + 0.5, results_all_reps['stay_duration_hist'] / np.sum(results_all_reps['stay_duration_hist']), 
    #            color = 'k', label = 'No COD')
    #     ax.set_yscale('log')
    #     plt.xlabel('Stay duration (trials)')
    #     plt.ylabel('Proportion')
    #     plt.legend()
    
    fig.show()
    
    # 3. -- Matching slope using Income VS Return (after Mar 4 2020 Foraging meeting)
    
    # fig = plt.figure(figsize=(15*1, 5*1))
    
    # gs = GridSpec(1,3, wspace=0.3, hspace=0.5, bottom=0.13) 
    
    # # === Pooled block-wise return_log_ratio across sessions
    # ax = fig.add_subplot(gs[0,1])
    
    # # Scatter plot
    # ax.plot(rtn_log_ratio, c_log_ratio, '.k')

    # # Get linear fit paras
    # # "a,b" in Corrado 2005, "slope" in Iigaya 2019
    # [a, a_CI95], [b, _],[r_square, p],[slope, slope_CI95] = results_all_reps['linear_fit_log_income_ratio'][0,:,:]

    # plt.xlabel('Blockwise log RETURN ratio')
    # plt.ylabel('Blockwise log choice ratio')
    # plt.plot([-4,4],[-4,4],'y--')
    # plt.title('POOLED across sessions')
    # # ax.set_aspect('equal','datalim')
    # plt.axis('square')
    
    # # === Matching slope using Incomd VS Return, session-wise
    # ax = fig.add_subplot(gs[0,2])
    
    # matching_slopes_income = results_all_reps['linear_fit_income_per_session'][0,:]
    # matching_slopes_return = results_all_reps['linear_fit_return_per_session'][0,:]
    # plt.plot(matching_slopes_income, matching_slopes_return,'r.') 
    
    # plt.plot([0,1],[0,1],'k--')       
    
    # plt.xlabel('Matching from INCOME log ratio')
    # plt.ylabel('Matching from RETURN log ratio')
    # plt.title('Block-wise for EACH SESSION (n_rep = %g)'% results_all_reps['n_reps'])
    # plt.axis('square')

  
def plot_para_scan(results_para_scan, para_to_scan, if_baited = True, p_reward_sum = 0.45, p_reward_pairs = None, **kwargs):
    
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

        # matching_slope = results_para_scan['linear_fit_log_income_ratio'][:,3,0]  # "Slope" in Iigaya 2019
        # matching_slope_CI95 = results_para_scan['linear_fit_log_income_ratio'][:,3,1]
        
        # For model competition / scan, it is unfair that I group all blocks over all sessions and then fit the line once.
        # Because this would lead to a very small slope_CI95 that may mask the high variability of matching slope due to extreme biases in never-explore regime.
        # I should compute a macthing slope for each session and then calculate the CI95 using the same way as foraging efficiency. 
        # paras_matching_slope = results_para_scan['linear_fit_per_session']
        # matching_slope = np.nanmean(paras_matching_slope, axis = 1)
        # matching_slope_CI95 = 1.96 * np.nanstd(paras_matching_slope, axis = 1) / np.sqrt(n_reps)
        
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
        random_result, ideal_result = get_baseline(if_baited, p_reward_sum, p_reward_pairs)
        
        xlim = [para_range[0], para_range[-1]]
        
        plt.plot(xlim, [random_result[0]]*2, 'k--')
        plt.fill_between(xlim, - np.diff(random_result), np.sum(random_result), alpha = 0.2, color ='black')
        plt.text(xlim[0], random_result[0], 'random')
        
        plt.plot(xlim, [ideal_result[0]]*2, 'k--')
        plt.fill_between(xlim, - np.diff(ideal_result), np.sum(ideal_result), alpha = 0.2, color ='black')
        plt.text(xlim[0], ideal_result[0], 'ideal')

        # -- 2. Matching slope vs para
        # ax = fig.add_subplot(gs[0,1])
        # plt.plot(para_range, matching_slope, '-')
        # plt.fill_between(para_range, matching_slope - matching_slope_CI95, matching_slope + matching_slope_CI95, label = '95% CI', alpha = 0.2)
        # plt.xlabel(para_name)
        # plt.ylabel('Matching slope (log ratio)')
        # ax.legend()
        # if if_log: ax.set_xscale('log')
        
        # -- 3. Foraging efficiency vs Matching slope
        # ax = fig.add_subplot(gs[0,2])
        # plt.plot(matching_slope, fe_mean, 'o-')
        # plt.xlabel('Matching slope (log ratio)')
        # plt.ylabel('Foraging efficiency')
        
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
        matching_slope = results_para_scan['linear_fit_log_income_ratio'][:,3,0].reshape(len(para_ranges[0]), len(para_ranges[1]))
        
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
        im = plt.imshow(fe_mean.T, interpolation=interp, cmap=cmap, vmin=0.7, vmax=1)
        
        ax.contour(ndimage.gaussian_filter(fe_mean.T, sigma=1), levels=20, colors='k')
        
        plt.xlabel(para_names[0])
        ax.set_xlim(-0.5, len(para_ranges[0])-0.5)
        ax.set_xticks(x_label_idx)
        ax.set_xticklabels(np.round(para_ranges[0][x_label_idx],2))
        plt.xticks(rotation=45)
        
        plt.ylabel(para_names[1])
        ax.set_ylim(-0.5, len(para_ranges[1])-0.5)
        ax.set_yticks(y_label_idx)
        ax.set_yticklabels(np.round(para_ranges[1][y_label_idx],2))
        
        plt.title(f'Foraging efficiency, max = {np.max(fe_mean)*100: .3g}%')
        
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
                
        # fig = px.imshow(fe_mean.T,
        #                 labels={'x': para_names[0], 'y': para_names[1], 'color': 'foraging eff.'},
        #                 x=[str(x) for x in np.round(para_ranges[0][:],2)],
        #                 y=[str(x) for x in np.round(para_ranges[1][:],2)],
        #                 aspect='auto',
        #                 origin='lower')
        
        # fig.update_layout(height=600, width=600, 
        #                   title_text= 'Forager = %s, n_repetitions = %g, %s' % (forager, n_reps, kwargs)
        #                   )
        # fig.show()
        
        # # -- 2. Matching slope
        # ax = fig.add_subplot(gs[0,1])
        # im = plt.imshow(matching_slope.T, interpolation=interp, cmap=cmap)
        
        # plt.xlabel(para_names[0])
        # ax.set_xlim(-0.5, len(para_ranges[0])-0.5)
        # ax.set_xticks(x_label_idx)
        # ax.set_xticklabels(np.round(para_ranges[0][x_label_idx],2))
        # plt.xticks(rotation=45)
        
        # plt.ylabel(para_names[1])
        # ax.set_ylim(-0.5, len(para_ranges[1])-0.5)
        # ax.set_yticks(y_label_idx)
        # ax.set_yticklabels(np.round(para_ranges[1][y_label_idx],2))
        
        # plt.title('Matching slope (log ratio)')
        
        # # create an axes on the right side of ax. The width of cax will be 5%
        # # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(im, cax=cax)
        
        # # -- 3. Foraging efficiency vs Matching slope
        # ax = fig.add_subplot(gs[0,2])
        # plt.plot(matching_slope.T, fe_mean.T, 'o')
        
        # plt.xlabel('Matching slope (log ratio)')
        # plt.ylabel('Foraging efficiency')
               
        
def plot_model_compet(model_compet_results, model_compet_settings, n_reps, baselines, if_baited = True, p_reward_sum = 0.45, p_reward_pairs = None):
    
    gs = GridSpec(1,1, top = 0.85, wspace = 0.3, bottom = 0.12)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(gs[0,0])
    
    # Let's fix the color coding
    colors = {'LossCounting':'C0', 'Sugrue2004':'C1', 'Corrado2005':'C2', 'Bari2019':'C3', 'Hattori2019':'C4', 'AmB1':'grey'}
        
    for this_result, this_setting in zip(model_compet_results, model_compet_settings):
    
        forager = this_setting['forager']
        para_to_scan = this_setting['para_to_scan']
        para_name, para_value = list(para_to_scan.items())[0]
        
        # === Plotting ===
        # Foraging efficiency vs Matching slope
        fe_mean, fe_CI95, matching_slope_mean, matching_slope_CI95 = this_result
        
        # Others
        order = np.argsort(para_value)
        plt.errorbar(matching_slope_mean[order], fe_mean[order], xerr = matching_slope_CI95[order], yerr = fe_CI95[order], 
                         label = '%s (%s)' % (forager, para_name), color = colors[forager])
        
        if forager in ['LossCounting','AmB1']:  # Reverse the order such that larger dot represents more exploration
            sizes = np.linspace(10,1,len(order))**2
        else:
            sizes = np.linspace(1,10,len(order))**2
            
        plt.scatter(matching_slope_mean[order], fe_mean[order], s = sizes, color = colors[forager])

        # Optimal value
        if forager not in ['AmB1']:
            plt.plot(matching_slope_mean[0], fe_mean[0], '*', markersize = 27, color = colors[forager])
        
        plt.xlabel('Matching slope')
        plt.ylabel('Foraging efficiency')

    
    # Theoretical upper bound
    plt.plot([0,1],[1,1],'k--')
    plt.text(0, 1, '100% = IdealpHatOptimal (theor.)', color = 'k')
 
    # Baseline foragers
    # baseline_models = ['IdealpHatOptimal','IdealpHatGreedy','pMatching','IdealpGreedy','Random']
    markers = ['*','*','s','^','X']
    sizes = [20,15,10,13,13]

    for bm_name, bm_eff, bm_ms, bm_marker, bm_size in zip(baselines[0],baselines[1],baselines[2],markers,sizes):
        if bm_name in ['Random','IdealpHatOptimal']:
            plt.fill_between([0,1], bm_eff[0] - bm_eff[1], bm_eff[0] + bm_eff[1], color = 'k', alpha = 0.2)

        plt.plot(bm_ms[0], bm_eff[0], bm_marker, color = 'grey', markeredgecolor = 'k', markersize=bm_size, label = '%s' % bm_name, zorder=-32)
        plt.errorbar(bm_ms[0], bm_eff[0], xerr = bm_ms[1], yerr = bm_eff[1], color = 'k')
        
    # Add a theoretical matching index for IdealOptimal
    # ms_IO_analytical = baselines[3]
    # plt.plot([ms_IO_analytical]*2, plt.ylim(),'k--')
    # plt.text(ms_IO_analytical, 0, 'theoretical matching index of IdealOptimal', rotation = 90)
       
    # ax.set_yticks([0,0.5,1])
    # plt.xlim([-0.02,1.02])
    # plt.ylim([0,1.2])
    ax.legend(fontsize = 10, loc = "lower right")
    
    plt.title('Model competition (if_baited = %s, p_rew_sum = %g\nn_reps = %g, p_override = %s)'% (if_baited, p_reward_sum, n_reps, p_reward_pairs))
    
        
        
        
        
        
        
        
    