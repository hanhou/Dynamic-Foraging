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

# Import my own modules
from foraging_testbed_models import Bandit
from foraging_testbed_plots import plot_all_sessions

methods = [ 
            # 'serial',
            'apply_async'     # Use multiprocessing.apply_async() for parallel computing (8~10x speed-up in my 8/16 I9-9900k)
          ]

LEFT = 0
RIGHT = 1
global_k_arm = 2
global_n_trials = 1000  
global_n_sessions = 100

def run_one_session(bandit):     
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
    temp = np.array([[-999]]) # -999 is to capture the first and the last stay
    changeover_position = np.where(np.diff(np.hstack((temp, bandit.choice_history, temp))))[1] 
    bandit.stay_durations = np.diff(changeover_position)
        
    return bandit   # For apply_async, in-place change is impossible since each worker uses "bandit" as 
                    # an independent local object. So I have to return "bandit" explicitly

def run_sessions_parallel(bandit, n_sessions = global_n_sessions, pool = ''):  
    # =============================================================================
    # Run simulations with the SAME bandit in serial or in parallel, repeating n_sessions
    # =============================================================================
   
    # Generate a series of deepcopys of bandit to make them independent!!
    bandits_all_sessions = []
    [bandits_all_sessions.append(copy.deepcopy(bandit)) for ss in range(n_sessions)]
        
    if pool == '':  # Serial computing (for debugging)
        start = time.time()         
        for ss, bb in tqdm(enumerate(bandits_all_sessions), total = n_sessions, desc='serial'):     # trange: progress bar. HH
            run_one_session(bb)     # There is no need to assign back the resulting bandit. (Modified inside run_one_session())
             
        print('--- serial finished in %g s ---' % (time.time()-start))
        
    else:    # Parallel computing using multiprocessing.apply_async()
        start = time.time()
        
        # Note the "," in (bb,). See here https://stackoverflow.com/questions/29585910/why-is-multiprocessings-apply-async-so-picky
        result_ids = [pool.apply_async(run_one_session, args = (bb,)) for bb in bandits_all_sessions]  
                        
        for ss, result_id in tqdm(enumerate(result_ids), total = n_sessions, desc='apply_async'):
            # For apply_async, the assignment is required, because the bb passed to the workers are local independent copys.
            bandits_all_sessions[ss] = result_id.get()  
            
        print('--- apply_async finished in %g s---' % (time.time()-start), flush=True)
        
    # =============================================================================
    # Compute summarizing results for all repetitions
    # =============================================================================
    results_all_reps = dict()
    stay_duration_hist_bins = np.arange(21) + 0.5
    
    # -- Session-wise --
    results_all_reps['foraging_efficiency_per_session'] = np.zeros(n_sessions)
    results_all_reps['stay_duration_hist'] = np.zeros(len(stay_duration_hist_bins)-1)
    
    # -- Block-wise --
    n_blocks_now = 0    
    for ss, bb in enumerate(bandits_all_sessions): n_blocks_now += bb.n_blocks
    results_all_reps['blockwise_stats'] = np.zeros([4, n_blocks_now])   # [choice_frac, reward_frac, log_choice_ratio, log_reward_ratio]
    
    # Loop over repetitions
    n_blocks_now = 0        
    for ss, bb in enumerate(bandits_all_sessions):
        # -- Session-wise --
        results_all_reps['foraging_efficiency_per_session'][ss] = bb.foraging_efficiency
        results_all_reps['stay_duration_hist'] += np.histogram(bb.stay_durations, bins = stay_duration_hist_bins)[0]
        
        # -- Block-wise --
        results_all_reps['blockwise_stats'][:, n_blocks_now : n_blocks_now + bb.n_blocks] =  \
            np.vstack([bb.blockwise_choice_fraction, 
                       bb.blockwise_reward_fraction, 
                       bb.blockwise_log_choice_ratio, 
                       bb.blockwise_log_reward_ratio])
        n_blocks_now += bb.n_blocks
        
        
    results_all_reps['foraging_efficiency'] = np.array([np.mean(results_all_reps['foraging_efficiency_per_session']),
                                                        np.std(results_all_reps['foraging_efficiency_per_session'])])
    # Basic info
    results_all_reps['n_sessions'] = n_sessions
    results_all_reps['n_trials'] = n_sessions * bandit.n_trials
    results_all_reps['n_blocks'] = n_blocks_now
    results_all_reps['description'] = bandits_all_sessions[0].description
    
    results_all_reps['example_session'] = bandits_all_sessions[0]

    # Plot summary statistics over repeated sessions for this bandit
    plot_all_sessions(results_all_reps) 
    
    return results_all_reps


def para_scan():
    
    effective_tau = 10
    step_size = 1 - np.exp(-1/effective_tau)  # Approximately 1/effective_tau, if tau >> 1
    
    # Generate a series of Bandit objects using different eps. HH
    # 'Random', 'AlwaysLEFT', 'IdealGreedy'; 'SuttonBartoRLBook', 'Sugrue2004', 'Corrado2005', 'Iigaya2019', 'Bari2019', 'Hattori2019'
    
    # bandit = Bandit(forager = 'SuttonBartoRLBook', epsilon = 0.2, step_sizes = step_size, if_baited = True, n_trials = global_n_trials)
    # bandit = Bandit(forager = 'Bari2019', epsilon = 0,  step_sizes = 0.1, forget_rate = 0.05, softmax_temperature = 0.4, if_baited = True, n_trials = global_n_trials)  
    bandit = Bandit(forager = 'Hattori2019', epsilon = 0,  step_sizes = [0.2, 0.1], forget_rate = 0.05, softmax_temperature = 0.4, if_baited = True, n_trials = global_n_trials)   
   
    # bandit = Bandit(forager = 'Sugrue2004', epsilon = 0.1,  taus = 15, if_baited = True, n_trials = global_n_trials) 
    # bandit = Bandit(forager = 'Corrado2005', epsilon = 0,  taus = [5, 15], w_taus = [0.9, 0.1], softmax_temperature = 3, if_baited = True, n_trials = global_n_trials)  
    # bandit = Bandit(forager = 'Iigaya2019', epsilon = 0,  taus = [15,100], w_taus = [0.9, 0.1], random_before_total_reward = 20, if_baited = True, n_trials = global_n_trials)
   
    
    # Run simulations, return best_action_counts and rewards. HH
    run_sessions_parallel(bandit, pool = pool)


if __name__ == '__main__':  # This line is essential for apply_async to run in Windows
    
    pool = ''
   
    if 'apply_async' in methods:
        n_worker = int(mp.cpu_count()/2)  # Optimal number = number of physical cores
        pool = mp.Pool(processes = n_worker)

    para_scan()
    
    if 'apply_async' in methods:
        pool.close()   # Just a good practice
        pool.join()
