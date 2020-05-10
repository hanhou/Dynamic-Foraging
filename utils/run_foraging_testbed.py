# =============================================================================
# Main function for simulating foraging_model_HH
# =============================================================================
# - Use apply_async() in multiprocessing for parallel computing (8~10x speed-up in my 8/16 I9-9900k)
# - Ray is another option, but unfortunately they don't support Windows yet.
#
# Feb 2020, Han Hou @ Houston
# Svoboda lab
# =============================================================================


import numpy as np
from tqdm import tqdm  # For progress bar. HH
import time
import multiprocessing as mp
import copy
import statsmodels.api as sm
import scipy.optimize as optimize

# Import my own modules
from utils.foraging_testbed_models import Bandit
from utils.foraging_testbed_plots import plot_all_reps, plot_para_scan, plot_model_compet

methods = [ 
            # 'serial',
            'apply_async'     # Use multiproce ssing.apply_async() for parallel computing (8~10x speed-up in my 8/16 I9-9900k)
          ]

LEFT = 0
RIGHT = 1
global_k_arm = 2
global_n_trials = 1000  
global_n_reps = 500


def run_one_session(bandit, para_scan = False, para_optim = False):     
    # =============================================================================
    # Simulate one session
    # =============================================================================
    bandit.reset()
    for t in range(bandit.n_trials):        
        # Loop: (Act --> Reward & New state)
        action = bandit.act()
        bandit.step(action)
        
    # =============================================================================
    # Compute results for this session
    # =============================================================================
    # -- 1. Foraging efficiency = Sum of actual rewards / Maximum number of rewards that could have been collected --
    bandit.actual_rewards = np.sum(bandit.reward_history)
    
    '''Don't know which one is the fairest''' #???
    # Method 1: Average of max(p_reward) 
    # bandit.maximum_rewards = np.sum(np.max(bandit.p_reward, axis = 0)) 
    # Method 2: Average of sum(p_reward).   [Corrado et al 2005: efficienty = 50% for choosing only one color]
    # bandit.maximum_rewards = np.sum(np.sum(bandit.p_reward, axis = 0)) 
    # Method 3: Maximum reward given the actual reward_available (one choice per trial constraint)
    # bandit.maximum_rewards = np.sum(np.any(bandit.reward_available, axis = 0))  # Equivalent to sum(max())
    # Method 4: Sum of all ever-baited rewards (not fair)  
    # bandit.maximum_rewards = np.sum(np.sum(bandit.reward_available, axis = 0))
    
    ''' Use ideal-p^-optimal'''
    # bandit.maximum_rewards = bandit.rewards_IdealpHatGreedy
    if not para_optim: 
        bandit.maximum_rewards = bandit.rewards_IdealpHatOptimal
    else:  # If in optimization, fast and good
        bandit.maximum_rewards = bandit.rewards_IdealpHatGreedy
        
    bandit.foraging_efficiency = bandit.actual_rewards / bandit.maximum_rewards
    
   
    if not para_optim:
         # -- 2. Blockwise statistics --
        temp_nans = np.zeros(bandit.n_blocks)
        temp_nans[:] = np.nan   # Better way?
        
        bandit.blockwise_choice_fraction = temp_nans.copy()
        bandit.blockwise_income_fraction = temp_nans.copy()
        bandit.blockwise_log_choice_ratio = temp_nans.copy()
        bandit.blockwise_log_income_ratio = temp_nans.copy()
        bandit.blockwise_log_return_ratio = temp_nans.copy()
        
        bandit.block_trans_time = np.cumsum(np.hstack([0,bandit.block_size]))
        
        for i_block in range(bandit.n_blocks):   # For each block in this session
            trial_range = np.r_[bandit.block_trans_time[i_block] : bandit.block_trans_time[i_block+1]]  # r_ trick
           
            choice_R = np.sum(bandit.choice_history[0,trial_range] == RIGHT)
            choice_L = np.sum(bandit.choice_history[0,trial_range] == LEFT)
            rew_R = np.sum(bandit.reward_history[RIGHT, trial_range])
            rew_L = np.sum(bandit.reward_history[LEFT, trial_range])
                    
            if (rew_R + rew_L):    # Non-zero total reward. Otherwise, leaves nan
                bandit.blockwise_choice_fraction[i_block] = choice_R / (choice_R + choice_L)
                bandit.blockwise_income_fraction[i_block] = rew_R / (rew_R + rew_L)
            
            if all([rew_R, rew_L, choice_R, choice_L]):   # All non-zero. Otherwise, leaves nan
                bandit.blockwise_log_choice_ratio[i_block] = np.log(choice_R / choice_L)
                bandit.blockwise_log_income_ratio[i_block] = np.log(rew_R / rew_L)
                
                # Let's try matching slope using 'RETURN' as well. (after Mar.4 2020 Foraging meeting)
                bandit.blockwise_log_return_ratio[i_block] = np.log((rew_R/choice_R)/(rew_L/choice_L))
                
        # -- 2.5 Matching for each session --
        # For model competition, it is unfair that I group all blocks over all sessions and then fit the line once.
        # Because this would lead to a very small slope_CI95 that may mask the high variability of matching slope due to extreme biases in never-explore regime.
        # I should compute a macthing slope for each session and then calculate the CI95 using the same way as foraging efficiency. 
        
        # --- Linear regression on log_ratios (for each session) ---
        c_fraction = bandit.blockwise_choice_fraction
        inc_fraction = bandit.blockwise_income_fraction
        
        c_log_ratio = bandit.blockwise_log_choice_ratio
        inc_log_ratio = bandit.blockwise_log_income_ratio 
        
        # Let's try matching slope using 'RETURN' as well. (after Mar.4 2020 Foraging meeting)
        rtn_log_ratio = bandit.blockwise_log_return_ratio
        
        
        bandit.linear_fit_income_per_session = np.nan
        bandit.linear_fit_return_per_session = np.nan
            
        if np.sum(~np.isnan(inc_log_ratio)) > 5:  # Use log_ratio
            
            # == 1. Income_log_ratio ==
            x = inc_log_ratio[~np.isnan(inc_log_ratio)]
            y = c_log_ratio[~np.isnan(c_log_ratio)]
            
            # Linear regression
            model = sm.OLS(y, sm.add_constant(x)).fit()
            
            intercept, a = model.params  # "a, b" in Corrado 2005
            # b = np.exp(intercept)
            intercept_CI95, a_CI95  = np.diff(model.conf_int(), axis=1)/2
            # r_square, p = (model.rsquared, model.pvalues)
            
            # From log ratio to fraction
            # slope = 4*a*b/(1+b)**2   #　"Slope" in Iigaya 2019: linear fitting of fractional choice vs fractional reward. By derivation  
            # slope_CI95 = a_CI95*4*b/(1+b)**2
           
            bandit.linear_fit_income_per_session = a  # Let's use slope from log ratio here, because I don't want to let the bias contaminate the ratio
            
            # == 2. Return_log_ratio ==            
            # Let's try matching slope using 'RETURN' as well. (after Mar.4 2020 Foraging meeting)
            x = rtn_log_ratio[~np.isnan(rtn_log_ratio)]
            y = c_log_ratio[~np.isnan(c_log_ratio)]
            
            # Linear regression
            model = sm.OLS(y, sm.add_constant(x)).fit()
            
            intercept, a = model.params  # "a, b" in Corrado 2005
            bandit.linear_fit_return_per_session = a  # Let's use slope from log ratio here, because I don't want to let the bias contaminate the ratio
            
        elif np.sum(~np.isnan(inc_fraction)) > 5:   # Otherwise, use fraction (only this way can we get a matching slope for ideal_greedy and forager with extreme bias)
            
            x = inc_fraction[~np.isnan(inc_fraction)]
            y = c_fraction[~np.isnan(c_fraction)]
            
            # Linear regression
            model = sm.OLS(y, sm.add_constant(x)).fit()
            
            try:
                intercept, a = model.params
                bandit.linear_fit_income_per_session = a  # Let's use slope from log ratio here, because I don't want to let the bias contaminate the ratio
            except:
                pass
              
        # -- 3. Stay duration --
        if not para_scan:
            temp = np.array([[-999]]) # -999 is to capture the first and the last stay
            changeover_position = np.where(np.diff(np.hstack((temp, bandit.choice_history, temp))))[1] 
            bandit.stay_durations = np.diff(changeover_position)
        
    return bandit   # For apply_async, in-place change is impossible since each worker uses "bandit" as 
                    # an independent local object. So I have to return "bandit" explicitly

def run_sessions_parallel(bandit, n_reps = global_n_reps, pool = '', para_optim = False, if_plot = True):  
    # =============================================================================
    # Run simulations with the same bandit (para_scan = 0) or a list of bandits (para_scan = 1), in serial or in parallel, repeating n_reps.
    # =============================================================================
    if isinstance(bandit, list):  # Whether we're doing a parameter scan.
        para_scan = 1
    else:
        para_scan = 0
        bandit = [bandit]   # For backward compatibility
   
    # Generate a series of deepcopys of bandit to make them independent!!
    bandits_all_sessions = []
    for bb in bandit:
        [bandits_all_sessions.append(copy.deepcopy(bb)) for ss in range(n_reps)]
        
    if pool == '':  # Serial computing (for debugging)
        start = time.time()         
        
        if not para_optim:  # Progress bar
            for ss, bb in tqdm(enumerate(bandits_all_sessions), total = len(bandits_all_sessions), desc='serial'):     # trange: progress bar. HH
                run_one_session(bb, para_scan, para_optim)     # There is no need to assign back the resulting bandit. (Modified inside run_one_session())
        else:
            for ss, bb in enumerate(bandits_all_sessions):     # trange: progress bar. HH
                run_one_session(bb, para_scan, para_optim)     # There is no need to assign back the resulting bandit. (Modified inside run_one_session())
            
                
        if not para_optim: print('--- serial finished in %g s ---' % (time.time()-start))
        
    else:    # Parallel computing using multiprocessing.apply_async()
        start = time.time()
        
        # Note the "," in (bb,). See here https://stackoverflow.com/questions/29585910/why-is-multiprocessings-apply-async-so-picky
        result_ids = [pool.apply_async(run_one_session, args = (bb, para_scan, para_optim)) for bb in bandits_all_sessions]  
                        
        if not para_optim:  # Progress bar
            for ss, result_id in tqdm(enumerate(result_ids), total = len(bandits_all_sessions), desc='apply_async'):
                # For apply_async, the assignment is required, because the bb passed to the workers are local independent copys.
                bandits_all_sessions[ss] = result_id.get()  
        else:
            for ss, result_id in enumerate(result_ids):
                # For apply_async, the assignment is required, because the bb passed to the workers are local independent copys.
                bandits_all_sessions[ss] = result_id.get()
            
        # if not para_optim: print('--- apply_async finished in %g s---' % (time.time()-start), flush=True)
        
    # =============================================================================
    # Compute summarizing results for all sessions
    # =============================================================================
    results_all_sessions = dict()
    stay_duration_hist_bins = np.arange(21) + 0.5
    n_unique_bandits = len(bandit)
    
    results_all_sessions['foraging_efficiency_per_session'] = np.zeros([n_unique_bandits, n_reps])
    
    if not (para_scan or para_optim):
        results_all_sessions['stay_duration_hist'] = np.zeros(len(stay_duration_hist_bins)-1)
        
        # if bandit[0].forager == 'IdealOptimal':
        #     results_all_sessions['matching_slope_IdealOptimal_theoretical_per_session'] = np.zeros(n_reps)
    
    if not para_optim:
        results_all_sessions['linear_fit_log_income_ratio'] = np.zeros([n_unique_bandits, 4, 2])
        results_all_sessions['linear_fit_log_income_ratio'][:] = np.nan
        results_all_sessions['linear_fit_income_per_session'] = np.zeros([n_unique_bandits, n_reps])
        results_all_sessions['linear_fit_return_per_session'] = np.zeros([n_unique_bandits, n_reps])
        
    # Loop over all unique bandits
    for unique_idx in range(n_unique_bandits):
        
        sessions_for_this_bandit = bandits_all_sessions [unique_idx * n_reps : (unique_idx + 1) * n_reps]
        
        n_blocks_now = 0    
        for ss, bb in enumerate(sessions_for_this_bandit): n_blocks_now += bb.n_blocks
        blockwise_stats_this_bandit = np.zeros([5, n_blocks_now])   # [choice_frac, reward_frac, log_choice_ratio, log_reward_ratio]
        
        # --- Loop over repetitions ---
        n_blocks_now = 0        
        for ss, bb in enumerate(sessions_for_this_bandit):
            # Session-wise
            results_all_sessions['foraging_efficiency_per_session'][unique_idx,ss] = bb.foraging_efficiency
            
            if not (para_scan or para_optim):
                results_all_sessions['stay_duration_hist'] += np.histogram(bb.stay_durations, bins = stay_duration_hist_bins)[0]
                
                # if bandit[0].forager == 'IdealOptimal':
                #     results_all_sessions['matching_slope_IdealOptimal_theoretical_per_session'][ss] = bb.matching_slope_IdealOptimal_theoretical

            if not para_optim:
                
                results_all_sessions['linear_fit_income_per_session'][unique_idx,ss] = bb.linear_fit_income_per_session # Add session-wise matching slope for model competition
                results_all_sessions['linear_fit_return_per_session'][unique_idx,ss] = bb.linear_fit_return_per_session # Matching slope from log_return_ratio

                blockwise_stats_this_bandit[:, n_blocks_now : n_blocks_now + bb.n_blocks] =  \
                    np.vstack([bb.blockwise_choice_fraction, 
                               bb.blockwise_income_fraction, 
                               bb.blockwise_log_choice_ratio, 
                               bb.blockwise_log_income_ratio,
                               bb.blockwise_log_return_ratio])  # Add the last one. Mar.5, 2020
                n_blocks_now += bb.n_blocks
            
        if not para_optim and not para_scan:   # For para_optim, we don't need this. For para_scan, I decided to use session-wise matching index.
            # --- Linear regression on log_ratios (moved here) ---
            c_log_ratio = blockwise_stats_this_bandit[2,:]
            inc_log_ratio = blockwise_stats_this_bandit[3,:]
                
            c_fraction = blockwise_stats_this_bandit[0,:]
            inc_fraction = blockwise_stats_this_bandit[1,:]
            
            if bandit[0].forager not in ['AlwaysLEFT','IdealpGreedy'] and np.sum(~np.isnan(inc_log_ratio)) > 10:  # Use log_ratio
                x = inc_log_ratio[~np.isnan(inc_log_ratio)]
                y = c_log_ratio[~np.isnan(c_log_ratio)]
                
                # Linear regression
                model = sm.OLS(y, sm.add_constant(x)).fit()
                
                intercept, a = model.params  # "a, b" in Corrado 2005
                b = np.exp(intercept)
                intercept_CI95, a_CI95  = np.diff(model.conf_int(), axis=1)/2
                r_square, p = (model.rsquared, model.pvalues)
                
                # From log ratio to fraction
                slope = 4*a*b/(1+b)**2   #　"Slope" in Iigaya 2019: linear fitting of fractional choice vs fractional reward. By derivation  
                slope_CI95 = a_CI95*4*b/(1+b)**2
               
                results_all_sessions['linear_fit_log_income_ratio'][unique_idx,:,:] = [a, a_CI95], [b, np.nan],[r_square, p[1]],[slope, slope_CI95]
                
            elif np.sum(~np.isnan(inc_fraction)) > 5:   # Otherwise, use fraction (only this way can we get a matching slope for ideal_greedy and forager with extreme bias)
            
                x = inc_fraction[~np.isnan(inc_fraction)]
                y = c_fraction[~np.isnan(c_fraction)]
                
                # Linear regression
                model = sm.OLS(y, sm.add_constant(x)).fit()
                
                try:
                    intercept, a = model.params
                    intercept_CI95, a_CI95  = np.diff(model.conf_int(), axis=1)/2
                    results_all_sessions['linear_fit_log_income_ratio'][unique_idx,:,:] = [np.nan, np.nan], [np.nan, np.nan],[np.nan, np.nan],[a, a_CI95]
                except:
                    pass
            
    if not para_scan:    
        results_all_sessions['foraging_efficiency'] = np.array([np.mean(results_all_sessions['foraging_efficiency_per_session']),
                                                      1.96 * np.std(results_all_sessions['foraging_efficiency_per_session'])/np.sqrt(n_reps)])
        
        results_all_sessions['blockwise_stats'] = blockwise_stats_this_bandit # We need this only when not para_scan
        
        # if bandit[0].forager == 'IdealOptimal':        
        #     results_all_sessions['matching_slope_IdealOptimal_theoretical'] = np.mean(results_all_sessions['matching_slope_IdealOptimal_theoretical_per_session'])

    # Basic info
    results_all_sessions['n_reps'] = n_reps
    results_all_sessions['forager'] = bandits_all_sessions[0].forager
    results_all_sessions['if_baited'] = bandits_all_sessions[0].if_baited
    results_all_sessions['p_reward_sum'] = bandits_all_sessions[0].p_reward_sum
    results_all_sessions['p_reward_pairs'] = bandits_all_sessions[0].p_reward_pairs
    
    # For runlength_anlaysis_Lau
    results_all_sessions['choice_history'] = np.hstack([bb.choice_history for bb in bandits_all_sessions])
    results_all_sessions['p_reward'] = np.hstack([bb.p_reward for bb in bandits_all_sessions])
    
    # If not in para_scan, plot summary statistics over repeated sessions for the SAME bandit
    if if_plot and not (para_scan or para_optim):
        results_all_sessions['n_trials'] = n_reps * bandit[0].n_trials
        results_all_sessions['n_blocks'] = n_blocks_now
        results_all_sessions['description'] = bandits_all_sessions[0].description
        results_all_sessions['example_session'] = bandits_all_sessions[0]
        
        plot_all_reps(results_all_sessions) 
    
    if not para_optim:
        return results_all_sessions
    else:  # if we are in automatica parameter optimization, we only care about the foraging efficiency
        return results_all_sessions['foraging_efficiency'][0] 


#%% =============================================================================
#  1-D or 2-D manual parameter scan
# =============================================================================
def para_scan(forager, para_to_scan, n_reps = global_n_reps, pool = '', if_plot = True, if_baited = True, p_reward_sum = 0.45, p_reward_pairs = None, **kwargs):
    
    # == Turn para_to_scan into list of Bandits ==
    n_nest = len(para_to_scan)
    bandits_to_scan = []
    
    if n_nest == 1:
        para_name, para_range = list(para_to_scan.items())[0]
        
        for pp in para_range:
            kwargs_all = {**{para_name:pp}, **kwargs}   # All parameters
            bandits_to_scan.append(Bandit(forager = forager, if_baited = if_baited,  p_reward_sum = p_reward_sum, p_reward_pairs = p_reward_pairs, **kwargs_all))   # Append to the list
            
    elif n_nest == 2:
        para_names = list(para_to_scan.keys())
        para_ranges = list(para_to_scan.values())
        
        for pp_1 in para_ranges[0]:
            for pp_2 in para_ranges[1]:
                
                if forager == 'Hattori2019':    # Stupid workaround...
                    kwargs_all = {'step_sizes': [pp_1, pp_2], **kwargs}
                    
                else:   
                    kwargs_all = {**{para_names[0]: pp_1, para_names[1]: pp_2}, **kwargs}
                    
                bandits_to_scan.append(Bandit(forager = forager, if_baited = if_baited, p_reward_sum = p_reward_sum, p_reward_pairs = p_reward_pairs, **kwargs_all))   # Append to the list
            
    results_para_scan = run_sessions_parallel(bandits_to_scan, n_reps = n_reps, pool = pool)
    if if_plot: plot_para_scan(results_para_scan, para_to_scan, if_baited = if_baited, p_reward_sum = p_reward_sum, p_reward_pairs = p_reward_pairs, **kwargs)
            
    return results_para_scan


#%% =============================================================================
#  Automatic parameters optimization (for performance)
# =============================================================================

def generate_kwargs(forager, opti_names, opti_value):  # Helper function for parameter intepretation
    
    if forager == 'Corrado2005':  # Special workarounds
        kwargs_all = {'forager': 'Corrado2005', 'taus': opti_value[0:2], 'w_taus': [1-opti_value[2], opti_value[2]], 'softmax_temperature': opti_value[3]}
    
    elif forager == 'Corrado2005_fixW':
        kwargs_all = {'forager': 'Corrado2005', 'taus': opti_value[0:2], 'w_taus': [0.33, 0.67], 'softmax_temperature': opti_value[2]}
    
    elif forager == 'Hattori2019':
        kwargs_all = {'forager': 'Hattori2019', 'step_sizes': opti_value[0:2], 'forget_rate': opti_value[2], 'softmax_temperature': opti_value[3]}

    else:
        kwargs_all = {'forager': forager}
        for (nn, vv) in zip(opti_names, opti_value):
            kwargs_all = {**kwargs_all, nn:vv}
            
    return kwargs_all
            
def score_func(opti_value, *argss):
        
    # Arguments interpretation
    forager, opti_names, n_reps_per_iter, if_baited, p_reward_sum, p_reward_pairs, if_varying_amplitude, pool = argss
    kwargs_all = generate_kwargs(forager, opti_names, opti_value)
        
    # Run simulation
    bandit = Bandit(**kwargs_all, if_baited = if_baited, p_reward_sum = p_reward_sum, 
                    p_reward_pairs = p_reward_pairs, p_reward_seed_override = 20200303, 
                    if_varying_amplitude = if_varying_amplitude,
                    if_para_optim = True)  # The same reward schedule for fair comparison
    score = - run_sessions_parallel(bandit, n_reps = n_reps_per_iter, pool = pool, para_optim = True)  # Negative efficiency as cost function
    
    # print(np.round(opti_value,4), score, '\n')
    
    return score


def para_optimize(forager, n_reps_per_iter = 200, opti_names = '', bounds = '', pool = '', 
                  if_baited = True, p_reward_sum = 0.45, p_reward_pairs = None, if_varying_amplitude = False):
    
    start = time.time()
    
    # Define parameters to optimize and their bounds    
    if opti_names == '' or bounds == '':  # Could be override
        if forager == 'LossCounting':
            opti_names = ['loss_count_threshold_mean','loss_count_threshold_std']
            bounds = optimize.Bounds([0,0],[100,10])
            
        elif forager == 'Sugrue2004':
            opti_names = ['taus','epsilon']
            bounds = optimize.Bounds([1,0],[100,1])
    
        elif forager == 'Bari2019':
            opti_names = ['step_sizes','forget_rate','softmax_temperature']
            bounds = optimize.Bounds([0.01,0,0.01],[0.5,0.2,1])
            
        elif forager == 'Corrado2005':
            opti_names = ['tau1', 'tau2', 'w2', 'softmax_temperature']
            bounds = optimize.Bounds([1,10,0,0.1],[10,50,1,1])
            
        elif forager == 'Corrado2005_fixW':
            opti_names = ['tau1', 'tau2', 'softmax_temperature']
            bounds = optimize.Bounds([1,10,0.1],[10,50,1])
    
        elif forager == 'Hattori2019':
            opti_names = ['step_size_unrew', 'step_size_rew', 'forget_rate', 'softmax_temperature']
            bounds = optimize.Bounds([0.01,0.01, 0, 0.1],[0.5, 0.5, 0.5, 1])
            
        elif forager == 'PatternMelioration':
            opti_names = ['step_sizes', 'pattern_meliorate_threshold']
            bounds = optimize.Bounds([0.01, 0.01],[1, 1])

        # elif forager == 'PatternMelioration_softmax':
        #     opti_names = ['step_sizes', 'pattern_meliorate_softmax_temp', 'pattern_meliorate_softmax_max_step']
        #     bounds = optimize.Bounds([0.01, 0.01, 1],[1, 1, 10])

        elif forager == 'PatternMelioration_softmax':
            opti_names = ['step_sizes', 'pattern_meliorate_softmax_temp']
            bounds = optimize.Bounds([0.01, 0.01],[1, 1])
            
        elif forager == 'FullStateQ_softmax':
            opti_names = ['step_sizes', 'softmax_temperature', 'discount_rate', 'max_run_length']
            bounds = optimize.Bounds([0.005, 0.01, 0, 2],[1, 1, 1, 20])
            
        elif forager == 'FullStateQ_epsilon':
            opti_names = ['step_sizes', 'epsilon', 'discount_rate', 'max_run_length']
            bounds = optimize.Bounds([0.005, 0.01, 0, 2],[1, 1, 1, 20])

        
    # Parameter optimization with DE    
    opti_para = optimize.differential_evolution(func = score_func, 
                                                args = (forager, opti_names, n_reps_per_iter, if_baited, p_reward_sum, p_reward_pairs, 
                                                        if_varying_amplitude, pool), 
                                                bounds = bounds, 
                                                workers = 1, disp=True, strategy = 'best1bin',
                                                mutation=(0.5, 1), recombination = 0.7, popsize = 20)

    # Rerun using the optimized parameters
    kwargs_all = generate_kwargs(forager, opti_names, opti_para.x)
        
    bandit = Bandit(if_baited = if_baited, p_reward_sum = p_reward_sum, p_reward_pairs = p_reward_pairs, **kwargs_all)
    run_sessions_parallel(bandit, n_reps = 500, pool = pool)
                          
    print(opti_para)
    print(opti_names)
    
    print('--- para_optimize finished in %g s ---' % (time.time()-start))
    
    return opti_para

#%% =============================================================================
#   Model competition (for performance, NOT model comparison for fitting data) 
# ===============================================================================
def model_compet(model_compet_settings, n_reps = 200, pool = '', if_baited = True, p_reward_sum = 0.45, p_reward_pairs = None):
    
    model_compet_results = []   # Foraging efficiency mean
    
    for this_setting in model_compet_settings:
        
        forager = this_setting['forager']
        para_to_scan = this_setting['para_to_scan']
        para_to_fix = this_setting['para_to_fix']
    
        # Run simulation
        results_para_scan = para_scan(forager, para_to_scan, **para_to_fix , if_baited = if_baited, p_reward_sum = p_reward_sum, p_reward_pairs = p_reward_pairs, n_reps = n_reps, pool = pool, if_plot = False)
        
        # Fetch data
        paras_foraging_efficiency = results_para_scan['foraging_efficiency_per_session']
        fe_mean = np.mean(paras_foraging_efficiency, axis = 1)
        fe_CI95 = 1.96 * np.std(paras_foraging_efficiency, axis = 1) / np.sqrt(n_reps)

        # matching_slope = results_para_scan['linear_fit_log_income_ratio'][:,3,0]  # "Slope" in Iigaya 2019
        # matching_slope_CI95 = results_para_scan['linear_fit_log_income_ratio'][:,3,1]
        
        # For model competition, it is unfair that I group all blocks over all sessions and then fit the line once.
        # Because this would lead to a very small slope_CI95 that may mask the high variability of matching slope due to extreme biases in never-explore regime.
        # I should compute a macthing slope for each session and then calculate the CI95 using the same way as foraging efficiency. 
        paras_matching_slope = results_para_scan['linear_fit_income_per_session']
        ms_mean = np.nanmean(paras_matching_slope, axis = 1)
        ms_CI95 = 1.96 * np.nanstd(paras_matching_slope, axis = 1) / np.sqrt(n_reps)

        # Cache data
        model_compet_results.append(np.vstack((fe_mean, fe_CI95, ms_mean, ms_CI95)))
        
    # Run baseline models: random, ideal_greedy, and ideal-p^-optimal
    baseline_models = ['IdealpHatOptimal','IdealpHatGreedy','pMatching','IdealpGreedy','Random']
    baseline_eff = []
    baseline_ms = []
    
    for bm in baseline_models:
        bandit = Bandit(forager = bm, if_baited = if_baited, p_reward_sum = p_reward_sum, p_reward_pairs = p_reward_pairs)
        results = run_sessions_parallel(bandit, n_reps = n_reps, if_plot = False, pool = pool)
        
        paras_matching_slope = results['linear_fit_income_per_session']
        ms_mean = np.nanmean(paras_matching_slope, axis = 1)
        ms_CI95 = 1.96 * np.nanstd(paras_matching_slope, axis = 1) / np.sqrt(n_reps)

        baseline_eff.append(results['foraging_efficiency'])
        baseline_ms.append([ms_mean, ms_CI95])
        
        # if bm == 'IdealOptimal':
        #     ms_IO_analytical = results['matching_slope_IdealOptimal_theoretical']   # Analytical matching slope of IdealOptimal
    
    plot_model_compet(model_compet_results, model_compet_settings, n_reps, 
                      [baseline_models, baseline_eff, baseline_ms], # , ms_IO_analytical], 
                      if_baited = if_baited, p_reward_sum = p_reward_sum, p_reward_pairs = p_reward_pairs)
    
    
#%%
def sandro():
    
    softmax_temperatures = np.power(10, np.linspace(-1.5,0,20 ))
    eff_matching = np.zeros([4,len(softmax_temperatures)])
    
    model_compet_results = []
    
    n_reps_per_iter = 200
    n_reps_run = 500
    
    for n,stst in enumerate(softmax_temperatures):
        
        forager = 'Bari2019'
        opti_names = ['step_sizes','forget_rate','softmax_temperature']
        bounds = optimize.Bounds([0.01,0,stst],[0.5,0.2,stst])  # Fix softmax_temperature
        
        # Parameter optimization with DE    
        opti_para = optimize.differential_evolution(func = score_func, args = (forager, opti_names, n_reps_per_iter, True, False, pool), bounds = bounds, 
                                                    workers = 1, disp = True, strategy = 'best1bin',
                                                    mutation=(0.5, 1), recombination = 0.7, popsize = 20, maxiter = 1000)
        
        # Rerun using the optimized parameters
        kwargs_all = generate_kwargs(forager, opti_names, opti_para.x)
        bandit = Bandit(**kwargs_all)
        results_all_sessions = run_sessions_parallel(bandit, n_reps = n_reps_run, pool = pool, if_plot = False)
        
        # Fetch data
        paras_foraging_efficiency = results_all_sessions['foraging_efficiency_per_session']
        fe_mean = np.mean(paras_foraging_efficiency, axis = 1)
        fe_CI95 = 1.96 * np.std(paras_foraging_efficiency, axis = 1) / np.sqrt(n_reps_run)
        paras_matching_slope = results_all_sessions['linear_fit_income_per_session']
        ms_mean = np.nanmean(paras_matching_slope, axis = 1)
        ms_CI95 = 1.96 * np.nanstd(paras_matching_slope, axis = 1) / np.sqrt(n_reps_run)
        
        eff_matching[:,n] = [fe_mean, fe_CI95, ms_mean, ms_CI95]
        
  
    model_compet_results = [eff_matching]
    model_compet_settings = {'forager': forager, 
                                'para_to_scan': {'softmax_temperature': softmax_temperatures}, 
                            },

    # Run baseline models: random, ideal_greedy, and ideal-p^-optimal
    baseline_models = ['IdealpHatOptimal','IdealpHatGreedy','pMatching','IdealpGreedy','Random']
    baseline_eff = []
    baseline_ms = []
    
    for bm in baseline_models:
        bandit = Bandit(forager = bm)
        results = run_sessions_parallel(bandit, n_reps = n_reps_run, if_plot = False, pool = pool)
        
        paras_matching_slope = results['linear_fit_income_per_session']
        ms_mean = np.nanmean(paras_matching_slope, axis = 1)
        ms_CI95 = 1.96 * np.nanstd(paras_matching_slope, axis = 1) / np.sqrt(n_reps_run)

        baseline_eff.append(results['foraging_efficiency'])
        baseline_ms.append([ms_mean, ms_CI95])

    plot_model_compet(model_compet_results, model_compet_settings, n_reps_run, 
                  [baseline_models, baseline_eff, baseline_ms], 
                  )

    
#%%   
if __name__ == '__main__':  # This line is essential for apply_async to run in Windows
    
    pool = ''
   
    if 'apply_async' in methods:
        n_worker = int(mp.cpu_count()/2)  # Optimal number = number of physical cores
        pool = mp.Pool(processes = n_worker)
        
    #%% =============================================================================
    #     Play with the model manually
    # =============================================================================
    
    # 'Random', 'AlwaysLEFT', 'IdealpGreedy'; 'SuttonBartoRLBook', 'Sugrue2004', 'Corrado2005', 'Iigaya2019', 'Bari2019', 'Hattori2019'
    
    # bandit = Bandit('Random', n_trials = global_n_trials)   
    # bandit = Bandit('AlwaysLEFT', n_trials = global_n_trials)
    # bandit = Bandit('IdealpGreedy', n_trials = global_n_trials)
    
    # bandit = Bandit('LossCounting', loss_count_threshold_mean = 3, loss_count_threshold_std = 0)   # Loss counting
    # bandit = Bandit('LossCounting', loss_count_threshold_mean = 1, loss_count_threshold_std = 0)   # Win-stay-loss-shift
    # bandit = Bandit('LossCounting', loss_count_threshold_mean = 0, loss_count_threshold_std = 0)   # Always switching
    # bandit = Bandit('LossCounting', loss_count_threshold_mean = np.inf, loss_count_threshold_std = 0)   # Always One Side
    
    
    # bandit = Bandit(forager = 'Sugrue2004', taus = 8.34597217, epsilon = 0.24859973, n_trials = global_n_trials) 
    # bandit = Bandit(forager = 'Corrado2005', taus = [3, 15], w_taus = [0.7, 0.3], softmax_temperature = 0.4, epsilon = 0, n_trials = global_n_trials) 
    # bandit = Bandit(forager = 'Iigaya2019', taus = [5,10000], w_taus = [0.7, 0.3], epsilon = 0.1, n_trials = global_n_trials) 
    
    # bandit = Bandit(forager = 'SuttonBartoRLBook',step_sizes = 0.1, epsilon = 0.2, n_trials = global_n_trials)
    # bandit = Bandit(forager = 'Bari2019', step_sizes = 0.28768228, forget_rate = 0.01592382, softmax_temperature = 0.37121355,  epsilon = 0, n_trials = global_n_trials)
    # bandit = Bandit(forager = 'Hattori2019', epsilon = 0,  step_sizes = [0.2, 0.1], forget_rate = 0.05, softmax_temperature = 0.4, 
    #                 if_varying_amplitude = True, n_trials = global_n_trials)   
 
    # bandit = Bandit(forager = 'IdealpHatOptimal')
    # bandit = Bandit(forager = 'pMatching')
    # bandit = Bandit(forager = 'AmB1', m_AmB1=5)
    
    # bandit = Bandit(forager = 'PatternMelioration', step_sizes = 0.2, pattern_meliorate_threshold = 0.1, block_size_mean = 80, 
    #                 if_varying_amplitude = True, n_trials = global_n_trials)
    # results_all_sessions = run_sessions_parallel(bandit, n_reps = 500, pool = pool)

    # bandit = Bandit(forager = 'PatternMelioration_softmax', step_sizes = 0.1477, pattern_meliorate_softmax_temp = 0.1781, 
    #                 # pattern_meliorate_softmax_max_step = 4.6,
    #                 if_varying_amplitude = False, block_size_mean = 80, n_trials = global_n_trials)
    
    # bandit = Bandit(forager = 'FullStateQ_softmax', block_size_mean = 200, n_trials = 1000, if_baited = True, p_reward_pairs = [[0.3, 0.05]], 
    #                 step_sizes = 0.1, discount_rate = 0.6, max_run_length = 15, softmax_temperature = 0.1, 
    #                 if_plot_Q = False, if_varying_amplitude = True )
    
    # # bandit.simulate()
    
    # # results_all_sessions = run_sessions_parallel(bandit, n_reps = 1, pool = '')  # For debugging, use this.
    
    # # # ['step_sizes', 'softmax_temperature', 'discount_rate', 'max_run_length']
    # # # [ 0.50085521,  0.08830155,  0.11788019, 12.85673119]

    # results_all_sessions = run_sessions_parallel(bandit, n_reps = 50, pool = pool)

    # # # # === Runlength analysis for PatternMelioration ===
    # choice_history = results_all_sessions['choice_history']
    # p_reward = results_all_sessions['p_reward']
    
    # from utils.plot_mice import analyze_runlength_Lau2005, plot_runlength_Lau2005
    # run_length_Lau = analyze_runlength_Lau2005(choice_history, p_reward, block_partitions = [70,70])
    # plot_runlength_Lau2005(run_length_Lau, [70,70]);


    #%% =============================================================================
    #     Parameter scan (1-D or 2-D)
    # =============================================================================
    # 1-D
        
    # -- Figure 2C in Sugrue et al., 2004
    # para_to_scan = {'taus': np.power(2, np.linspace(0,8,15)),
    #                 # 'epsilon': np.linspace(0,0.5,6),
    #                 }
    # results_para_scan = para_scan('Sugrue2004', para_to_scan, epsilon = 0.15, n_reps = 100, pool = pool)
    
    # -- Figure 3 d and e of Iigaya et al, 2019
    # w_taus = [[1-w_slow, w_slow] for w_slow in np.linspace(0,1,10)]
    # para_to_scan = {'w_taus': w_taus
    #                 }
    # results_para_scan = para_scan('Iigaya2019', para_to_scan, taus = [2,1000],  epsilon = 0.1, n_reps = 100, pool = pool)
    
    #  2-D
    # -- Sugrue et al., 2004 in 2D
    # para_to_scan = {'taus': np.power(2, np.linspace(0,8,10)),
    #                 'epsilon': np.linspace(0,0.6,10),
    #                 }
    # results_para_scan = para_scan('Sugrue2004', para_to_scan, n_reps = 100, pool = pool)
    
    # -- Figure 11B of Corrado et al 2005
    # taus = [[2, tau_2] for tau_2 in np.power(2, np.linspace(0,8,10))]
    # para_to_scan = {'softmax_temperature': np.power(10, np.linspace(-1.5,0,10)),
    #                 'taus': taus,
    #                 }
    # results_para_scan = para_scan('Corrado2005', para_to_scan, w_taus = [0.33, 0.67],  epsilon = 0, n_reps = 100, pool = pool)
 
    # --PatternMelioration in 2D
    # para_to_scan = {'step_sizes': np.power(10, np.linspace(-2,0,10)),
    #                 'pattern_meliorate_threshold': np.power(10, np.linspace(-2,0,10)),
    #                 }
    # results_para_scan = para_scan('PatternMelioration', para_to_scan, n_reps = 50, pool = pool)
      

    # --PatternMelioration_softmax in 2D
    # para_to_scan = {'step_sizes': np.power(10, np.linspace(-1,0,10)),
    #                 'pattern_meliorate_softmax_temp': np.power(10, np.linspace(-1,0,10)),
    #                 }
    # results_para_scan = para_scan('PatternMelioration_softmax', para_to_scan, n_reps = 50, 
    #                               # pattern_meliorate_softmax_max_step = 2.5,
    #                               pool = pool)
           
    # --FullStateQ_softmax in 2D
    # para_to_scan = {'step_sizes': np.power(10, np.linspace(-1,0,15)),
    #                 'softmax_temperature': np.power(10, np.linspace(-1,0,15)),
    #                 }
    # results_para_scan = para_scan('FullStateQ_softmax', para_to_scan, n_reps = 100, 
    #                               max_run_length = 12.8567, discount_rate = 0.99, # softmax_temperature = 0.1,
    #                               pool = pool)
    
    #%% =============================================================================
    #   Automatic Parameter Optimization for Performance
    # =============================================================================
        
    # Special     
    # opti_para = para_optimize('LossCounting', n_reps_per_iter = 200, pool = pool)    # 85.4% @ [0.37879938, 0.18915971]
        
    # LNP-like
    # opti_para = para_optimize('Sugrue2004', n_reps_per_iter = 200, pool = pool)  # 81.5% @ [9.88406144, 0.313648  ]
    # opti_para = para_optimize('Corrado2005', n_reps_per_iter = 200, pool = pool)   #82.9% [ 6.17853872, 30.31342409,  0.04822465,  0.18704151]
    # opti_para = para_optimize('Corrado2005_fixW', n_reps_per_iter = 200, pool = pool) # 82.1% [ 3.79352945, 12.93486304,  0.22950269]
    
    # RL-like
    # opti_para = para_optimize('Bari2019', n_reps_per_iter = 200, pool = pool)  # 83.7% @ [0.37058271, 0.07003851, 0.27212561]
    # opti_para = para_optimize('Hattori2019', n_reps_per_iter = 200, pool = pool)  # 83.7% @ [0.39740558, 0.22740528, 0.11980517, 0.33762251] 

    # PatternMelioration 
    # opti_para = para_optimize('PatternMelioration', n_reps_per_iter = 200, pool = pool)    
    # [0.75390183, 0.67403303] for block-model-free;  
    # [0.86710063, 0.11670503] 94.4% for block-model-based-reset (reset pattern only);
    
    # opti_para = para_optimize('PatternMelioration_softmax', n_reps_per_iter = 200, pool = pool)
    # [0.69983055, 0.92476905, 4.65317582]  92.1%, sigmoid --> update step
    
    # opti_para = para_optimize('PatternMelioration_softmax', n_reps_per_iter = 200, pool = pool)
    #  [0.1477 0.1781]         softmax --> p --> m = floor(p/(1-p))

    # opti_para = para_optimize('FullStateQ_softmax', n_reps_per_iter = 200, pool =  pool)
    # ['step_sizes', 'softmax_temperature', 'discount_rate', 'max_run_length']
    # [ 0.50085521,  0.08830155,  0.11788019, 12.85673119]

    opti_para = para_optimize('FullStateQ_epsilon', if_varying_amplitude = True, n_reps_per_iter = 200, pool =  pool)

    # =============================================================================
    #   For higher total reward prob (sum = 0.8)
    # =============================================================================

    # Special     
    # opti_para = para_optimize('LossCounting', n_reps_per_iter = 200, p_reward_sum = 0.8, pool = pool)    # 76.6% [0.31005873, 0.08627981]
        
    # LNP-like
    # opti_para = para_optimize('Corrado2005', n_reps_per_iter = 200, p_reward_sum = 0.8, pool = pool) # 75.4% [4.16259724e+00, 3.61409637e+01, 1.71017422e-02, 2.47504000e-01]
    
    # RL-like
    # opti_para = para_optimize('Bari2019', n_reps_per_iter = 200, p_reward_sum = 0.8, pool = pool)  # 77.3% [0.40388374, 0.19951699, 0.22852593]
        
    # =============================================================================
    #   For extreme reward ratio (sum = 0.45, ratio = [0.9, 0])
    # =============================================================================
    # Special     
    # opti_para = para_optimize('LossCounting', n_reps_per_iter = 200, p_reward_pairs = [[0.45, 0]], pool = pool)    # 76.6% [0.31005873, 0.08627981]
        
    # LNP-like
    # opti_para = para_optimize('Corrado2005', n_reps_per_iter = 200, p_reward_pairs = [[0.45, 0]],  pool = pool) # 75.4% [4.16259724e+00, 3.61409637e+01, 1.71017422e-02, 2.47504000e-01]
    
    # RL-like
    # opti_para = para_optimize('Bari2019', n_reps_per_iter = 200, p_reward_pairs = [[0.45, 0]], pool = pool)  # 77.3% [0.40388374, 0.19951699, 0.22852593]
       

        
    #%% ===========================================================================
    #   Model Competition (1-d slice across the global optimum of each model)
    # =============================================================================
    
    # model_compet_settings = [
        
    #     {'forager': 'LossCounting', 
    #       'para_to_scan': {'loss_count_threshold_mean': np.hstack([0.37879938,np.power(2,np.linspace(0,6,13))])}, 
    #       'para_to_fix': {'loss_count_threshold_std': 0.18915971}}, 
               
    #     # {'forager': 'Sugrue2004', 
    #     #  'para_to_scan': {'taus': np.hstack([9.88406144, np.power(2, np.linspace(0,8,15))])},
    #     #  'para_to_fix':  {'epsilon': 0.313648}},
        
    #     # {'forager': 'Sugrue2004', 
    #     #   'para_to_scan': {'epsilon': np.hstack([0.313648, np.linspace(0,1,20)])},
    #     #   'para_to_fix':  {'taus': 9.88406144}},
         
    #     # {'forager': 'Corrado2005', 
    #     #  'para_to_scan': {'w_taus': [[1-w_slow, w_slow] for w_slow in np.hstack([0.04822465, np.linspace(0,1,15)])]}, 
    #     #  'para_to_fix':  {'taus':  [6.17853872, 30.31342409], 'softmax_temperature':  0.18704151}},
        
    #     # {'forager': 'Corrado2005', 
    #     #   'para_to_scan': {'softmax_temperature': np.hstack([0.18704151, np.power(10, np.linspace(-1.5,0,15))])}, 
    #     #   'para_to_fix':  {'taus':  [6.17853872, 30.31342409], 'w_taus': [1-0.04822465, 0.04822465]}},
         
    #     # {'forager': 'Bari2019', 
    #     #   'para_to_scan': {'step_sizes': np.hstack([0.37058271, np.power(10, np.linspace(-2,0,15))])}, 
    #     #   'para_to_fix':  {'forget_rate': 0.07003851, 'softmax_temperature': 0.27212561}},

    #     # {'forager': 'Bari2019', 
    #     #   'para_to_scan': {'softmax_temperature': np.hstack([0.27212561, np.power(10, np.linspace(-1.5,0,20))])}, 
    #     #   'para_to_fix':  {'forget_rate': 0.07003851, 'step_sizes': 0.37058271}},

         
    #     {'forager': 'Hattori2019', 
    #       'para_to_scan': {'softmax_temperature': np.hstack([0.33762251, np.power(10, np.linspace(-1.5,0,15))])}, 
    #       'para_to_fix':  {'forget_rate':  0.11980517, 'step_sizes': [0.39740558, 0.22740528]}},
        
    #     ]

    # model_compet(model_compet_settings, if_baited = True, n_reps = 100, pool = pool) 
        
       
        
    # =============================================================================
    #         
    # =============================================================================
                
    # model_compet_settings = [
    #     {'forager': 'LossCounting', 
    #       'para_to_scan': {'loss_count_threshold_mean': np.hstack([0.31005873, 0,np.power(2,np.linspace(0,6,10)), np.inf])}, 
    #       'para_to_fix': {'loss_count_threshold_std':  0.08627981}}, 
    #     # {'forager': 'Corrado2005', 
    #     #   'para_to_scan': {'softmax_temperature': np.hstack([0.24750, np.power(10, np.linspace(-4,0,10))])}, 
    #     #   'para_to_fix':  {'taus':  [4.16259724e+00, 3.61409637e+01], 'w_taus': [1-1.71017422e-02, 1.71017422e-02]}},
    #     # {'forager': 'Bari2019', 
    #     #   'para_to_scan': {'softmax_temperature': np.hstack([0.22852593, np.power(10, np.linspace(-1.5,0,10))])}, 
    #     #   'para_to_fix':  {'step_sizes': 0.40388374, 'forget_rate': 0.19951699}},
    #     {'forager': 'AmB1',
    #       'para_to_scan': {'m_AmB1': np.r_[1:20]},
    #       'para_to_fix':{}},
    #    ]

    # model_compet(model_compet_settings, n_reps = 10, if_baited = True, pool = pool) 

    # bandit = Bandit(forager = 'Corrado2005', taus = [3, 15], w_taus = [0.7, 0.3], softmax_temperature = 0.001, epsilon = 0, n_trials = global_n_trials) 
    # run_sessions_parallel(bandit, n_reps = 100, pool = pool)
          
        
    #%% Answer Sandro's question: what if optimizing all other parameters for each epsilon/sigma
    # sandro()

    
    
    #%% Clear up
    if pool != '':
        pool.close()   # Just a good practice
        pool.join()
