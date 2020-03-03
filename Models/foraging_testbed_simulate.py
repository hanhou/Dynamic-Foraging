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

# Import my own modules
from foraging_testbed_models import Bandit
from foraging_testbed_plots import plot_all_reps, plot_para_scan

methods = [ 
            # 'serial',
            'apply_async'     # Use multiprocessing.apply_async() for parallel computing (8~10x speed-up in my 8/16 I9-9900k)
          ]

LEFT = 0
RIGHT = 1
global_k_arm = 2
global_n_trials = 1000  
global_n_reps = 500

global pool

def run_one_session(bandit, para_scan = False):     
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
    bandit.actual_reward_rate = np.sum(bandit.reward_history) / bandit.n_trials
    
    '''Don't know which one is the fairest''' #???
    # bandit.maximum_reward_rate = np.mean(np.max(bandit.p_reward, axis = 0)) # Method 1: Average of max(p_reward) 
    bandit.maximum_reward_rate = np.mean(np.sum(bandit.p_reward, axis = 0)) # Method 2: Average of sum(p_reward).   [Corrado et al 2005: efficienty = 50% for choosing only one color]
    # bandit.maximum_reward_rate = np.sum(np.any(bandit.reward_available, axis = 0)) / bandit.n_trials  # Method 3: Maximum reward given the fixed reward_available (one choice per trial constraint) [Sugrue 2004???]
    # bandit.maximum_reward_rate = np.sum(np.sum(bandit.reward_available, axis = 0)) / bandit.n_trials  # Method 4: Sum of all ever-baited rewards (not fair)  

    bandit.foraging_efficiency = bandit.actual_reward_rate / bandit.maximum_reward_rate
    
    # -- 2. Blockwise statistics --
    temp_nans = np.zeros(bandit.n_blocks)
    temp_nans[:] = np.nan   # Better way?
    
    bandit.blockwise_choice_fraction = temp_nans.copy()
    bandit.blockwise_reward_fraction = temp_nans.copy()
    bandit.blockwise_log_choice_ratio = temp_nans.copy()
    bandit.blockwise_log_reward_ratio = temp_nans.copy()
    
    bandit.block_trans_time = np.cumsum(np.hstack([0,bandit.block_size]))
    
    for i_block in range(bandit.n_blocks):   # For each block in this session
        trial_range = np.r_[bandit.block_trans_time[i_block] : bandit.block_trans_time[i_block+1]]  # r_ trick
       
        choice_R = np.sum(bandit.choice_history[0,trial_range] == RIGHT)
        choice_L = np.sum(bandit.choice_history[0,trial_range] == LEFT)
        rew_R = np.sum(bandit.reward_history[RIGHT, trial_range])
        rew_L = np.sum(bandit.reward_history[LEFT, trial_range])
                
        if (rew_R + rew_L):    # Non-zero total reward. Otherwise, leaves nan
            bandit.blockwise_choice_fraction[i_block] = choice_R / (choice_R + choice_L)
            bandit.blockwise_reward_fraction[i_block] = rew_R / (rew_R + rew_L)
        
        if all([rew_R, rew_L, choice_R, choice_L]):   # All non-zero. Otherwise, leaves nan
            bandit.blockwise_log_choice_ratio[i_block] = np.log(choice_R / choice_L)
            bandit.blockwise_log_reward_ratio[i_block] = np.log(rew_R / rew_L)
            
    # -- 3. Stay duration --
    if not para_scan:
        temp = np.array([[-999]]) # -999 is to capture the first and the last stay
        changeover_position = np.where(np.diff(np.hstack((temp, bandit.choice_history, temp))))[1] 
        bandit.stay_durations = np.diff(changeover_position)
        
    return bandit   # For apply_async, in-place change is impossible since each worker uses "bandit" as 
                    # an independent local object. So I have to return "bandit" explicitly

def run_sessions_parallel(bandit, n_reps = global_n_reps, pool = ''):  
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
        for ss, bb in tqdm(enumerate(bandits_all_sessions), total = len(bandits_all_sessions), desc='serial'):     # trange: progress bar. HH
            run_one_session(bb, para_scan)     # There is no need to assign back the resulting bandit. (Modified inside run_one_session())
             
        print('--- serial finished in %g s ---' % (time.time()-start))
        
    else:    # Parallel computing using multiprocessing.apply_async()
        start = time.time()
        
        # Note the "," in (bb,). See here https://stackoverflow.com/questions/29585910/why-is-multiprocessings-apply-async-so-picky
        result_ids = [pool.apply_async(run_one_session, args = (bb, para_scan)) for bb in bandits_all_sessions]  
                        
        for ss, result_id in tqdm(enumerate(result_ids), total = len(bandits_all_sessions), desc='apply_async'):
            # For apply_async, the assignment is required, because the bb passed to the workers are local independent copys.
            bandits_all_sessions[ss] = result_id.get()  
            
        print('--- apply_async finished in %g s---' % (time.time()-start), flush=True)
        
    # =============================================================================
    # Compute summarizing results for all sessions
    # =============================================================================
    results_all_sessions = dict()
    stay_duration_hist_bins = np.arange(21) + 0.5
    n_unique_bandits = len(bandit)
    
    if not para_scan:
        results_all_sessions['stay_duration_hist'] = np.zeros(len(stay_duration_hist_bins)-1)
    
    
    results_all_sessions['foraging_efficiency_per_session'] = np.zeros([n_unique_bandits, n_reps])
    results_all_sessions['linear_fit_log_ratio'] = np.zeros([n_unique_bandits, 4, 2])
    results_all_sessions['linear_fit_log_ratio'][:] = np.nan
    
    # Loop over all unique bandits
    for unique_idx in range(n_unique_bandits):
        
        sessions_for_this_bandit = bandits_all_sessions [unique_idx * n_reps : (unique_idx + 1) * n_reps]
        
        n_blocks_now = 0    
        for ss, bb in enumerate(sessions_for_this_bandit): n_blocks_now += bb.n_blocks
        blockwise_stats_this_bandit = np.zeros([4, n_blocks_now])   # [choice_frac, reward_frac, log_choice_ratio, log_reward_ratio]
        
        # --- Loop over repetitions ---
        n_blocks_now = 0        
        for ss, bb in enumerate(sessions_for_this_bandit):
            # Session-wise
            results_all_sessions['foraging_efficiency_per_session'][unique_idx,ss] = bb.foraging_efficiency
            
            if not para_scan:
                results_all_sessions['stay_duration_hist'] += np.histogram(bb.stay_durations, bins = stay_duration_hist_bins)[0]
            
            # Block-wise
            blockwise_stats_this_bandit[:, n_blocks_now : n_blocks_now + bb.n_blocks] =  \
                np.vstack([bb.blockwise_choice_fraction, 
                           bb.blockwise_reward_fraction, 
                           bb.blockwise_log_choice_ratio, 
                           bb.blockwise_log_reward_ratio])
            n_blocks_now += bb.n_blocks
            
        # --- Linear regression on log_ratios (moved here) ---
        c_log_ratio = blockwise_stats_this_bandit[2,:]
        r_log_ratio = blockwise_stats_this_bandit[3,:]
            
        if bandit[0].forager not in ['AlwaysLEFT','IdealGreedy'] and np.sum(~np.isnan(r_log_ratio)) > 10:
            x = r_log_ratio[~np.isnan(r_log_ratio)]
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
           
            results_all_sessions['linear_fit_log_ratio'][unique_idx,:,:] = [a, a_CI95], [b, np.nan],[r_square, p[1]],[slope, slope_CI95]
        
    if not para_scan:    
        results_all_sessions['foraging_efficiency'] = np.array([np.mean(results_all_sessions['foraging_efficiency_per_session']),
                                                      1.96 * np.std(results_all_sessions['foraging_efficiency_per_session'])/np.sqrt(n_reps)])
        results_all_sessions['blockwise_stats'] = blockwise_stats_this_bandit # We need this only when not para_scan
        
    # Basic info
    results_all_sessions['n_reps'] = n_reps
    results_all_sessions['forager'] = bandits_all_sessions[0].forager
    
    # If not in para_scan, plot summary statistics over repeated sessions for the SAME bandit
    if not para_scan:
        results_all_sessions['n_trials'] = n_reps * bandit[0].n_trials
        results_all_sessions['n_blocks'] = n_blocks_now
        results_all_sessions['description'] = bandits_all_sessions[0].description
        results_all_sessions['example_session'] = bandits_all_sessions[0]
        
        plot_all_reps(results_all_sessions) 
    
    return results_all_sessions


def para_scan(forager, para_to_scan, n_reps = global_n_reps, pool = '', **kwargs):
    
    # == Turn para_to_scan into list of Bandits ==
    n_nest = len(para_to_scan)
    bandits_to_scan = []
    
    if n_nest == 1:
        para_name, para_range = list(para_to_scan.items())[0]
        
        for pp in para_range:
            kwargs_all = {**{para_name:pp}, **kwargs}   # All parameters
            bandits_to_scan.append(Bandit(forager = forager, **kwargs_all))   # Append to the list
            
    elif n_nest == 2:
        para_names = list(para_to_scan.keys())
        para_ranges = list(para_to_scan.values())
        
        for pp_1 in para_ranges[0]:
            for pp_2 in para_ranges[1]:
                kwargs_all = {**{para_names[0]: pp_1, para_names[1]: pp_2}, **kwargs}
                bandits_to_scan.append(Bandit(forager = forager, **kwargs_all))   # Append to the list
            
    results_para_scan = run_sessions_parallel(bandits_to_scan, n_reps = n_reps, pool = pool)
    plot_para_scan(results_para_scan, para_to_scan, **kwargs)
            
    return results_para_scan
   
if __name__ == '__main__':  # This line is essential for apply_async to run in Windows
    
    global pool
    pool = ''
   
    if 'apply_async' in methods:
        n_worker = int(mp.cpu_count()/2)  # Optimal number = number of physical cores
        pool = mp.Pool(processes = n_worker)
        
    # =============================================================================
    #     Play with the model manually
    # =============================================================================
    
    # 'Random', 'AlwaysLEFT', 'IdealGreedy'; 'SuttonBartoRLBook', 'Sugrue2004', 'Corrado2005', 'Iigaya2019', 'Bari2019', 'Hattori2019'
    
    # bandit = Bandit('Random', n_trials = global_n_trials)   
    # bandit = Bandit('AlwaysLEFT', n_trials = global_n_trials)
    # bandit = Bandit('IdealGreedy', n_trials = global_n_trials)
    
    # bandit = Bandit('LossCounting', loss_threshold_mean = 3, loss_threshold_std = 0)   # Loss counting
    bandit = Bandit('LossCounting', loss_threshold_mean = 1, loss_threshold_std = 0)   # Win-stay-loss-shift
    bandit = Bandit('LossCounting', loss_threshold_mean = 0, loss_threshold_std = 0)   # Always switching
    # bandit = Bandit('LossCounting', loss_threshold_mean = 1000000, loss_threshold_std = 1)   # Always One Side
    
    
    # bandit = Bandit(forager = 'Sugrue2004', taus = 15, epsilon = 0.15, n_trials = global_n_trials) 
    # bandit = Bandit(forager = 'Corrado2005', taus = [3, 15], w_taus = [0.7, 0.3], softmax_temperature = 0.2, epsilon = 0, n_trials = global_n_trials) 
    # bandit = Bandit(forager = 'Iigaya2019', taus = [5,10000], w_taus = [0.7, 0.3], epsilon = 0.1, n_trials = global_n_trials) 
    
    # bandit = Bandit(forager = 'SuttonBartoRLBook',step_sizes = 0.1, epsilon = 0.2, n_trials = global_n_trials)
    # bandit = Bandit(forager = 'Bari2019', step_sizes = 0.2, forget_rate = 0.05, softmax_temperature = 0.4,  epsilon = 0, n_trials = global_n_trials)
    # bandit = Bandit(forager = 'Hattori2019', epsilon = 0,  step_sizes = [0.2, 0.1], forget_rate = 0.05, softmax_temperature = 0.4, n_trials = global_n_trials)   
 
    run_sessions_parallel(bandit, n_reps = global_n_reps, pool = pool)

    # =============================================================================
    #     Parameter scan (1-D or 2-D)
    # =============================================================================
    # 1-D
        
    #%% -- Figure 2C in Sugrue et al., 2004
    # para_to_scan = {'taus': np.power(2, np.linspace(0,8,15)),
    #                 # 'epsilon': np.linspace(0,0.5,6),
    #                 }
    # results_para_scan = para_scan('Sugrue2004', para_to_scan, epsilon = 0.15, n_reps = 100, pool = pool)
    
    #%% -- Figure 3 d and e of Iigaya et al, 2019
    # w_taus = [[1-w_slow, w_slow] for w_slow in np.linspace(0,1,10)]
    # para_to_scan = {'w_taus': w_taus
    #                 }
    # results_para_scan = para_scan('Iigaya2019', para_to_scan, taus = [2,1000],  epsilon = 0.1, n_reps = 100, pool = pool)
    
    #%% 2-D
    #%% -- Sugrue et al., 2004 in 2D
    # para_to_scan = {'taus': np.power(2, np.linspace(0,8,10)),
    #                 'epsilon': np.linspace(0,0.6,10),
    #                 }
    # results_para_scan = para_scan('Sugrue2004', para_to_scan, n_reps = 100, pool = pool)
    
    #%% -- Figure 11B of Corrado et al 2005
    # taus = [[2, tau_2] for tau_2 in np.power(2, np.linspace(0,8,10))]
    # para_to_scan = {'softmax_temperature': np.power(10, np.linspace(-1.5,0,10)),
    #                 'taus': taus,
    #                 }
    # results_para_scan = para_scan('Corrado2005', para_to_scan, w_taus = [0.33, 0.67],  epsilon = 0, n_reps = 100, pool = pool)

    if pool != '':
        pool.close()   # Just a good practice
        pool.join()
