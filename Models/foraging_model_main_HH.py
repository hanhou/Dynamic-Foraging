# =============================================================================
# Main function for simulating foraging_model_HH
# =============================================================================
# - Use apply_async() in multiprocessing for parallel computing (10x speed-up in my 8/16 I9-9900k)
#
# Han Hou @ Houston, Feb 12 2020
# Svoboda lab
# =============================================================================


import numpy as np
from tqdm import tqdm  # For progress bar. HH
import time
import multiprocessing as mp
import copy

# Import my own modules
from foraging_model_HH import Bandit
from foraging_model_plots_HH import plot_all_sessions

LEFT = 0
RIGHT = 1
global_k_arm = 2
global_n_trials = 700  # To cope with the one-argument limitation of map/imap
global_n_sessions = 10

def run_one_session(bandit):     
# =============================================================================
# Make one-run independently
# =============================================================================
    
    # === Simulate one run ===
    bandit.reset()
    for t in range(bandit.n_trials):        
        # Loop: (Act --> Reward & New state)
        action = bandit.act()
        bandit.step(action)
        
    # === Summarize results for this run ===
    # -- 1. Foraging efficiency = Sum of actual rewards / Maximum number of rewards that could have been collected --
    bandit.actual_reward_rate = np.sum(bandit.reward_history) / bandit.n_trials
    
    '''Don't know which one is better'''
    # bandit.maximum_reward_rate = np.mean(np.max(bandit.p_reward, axis = 0)) #??? Method 1: Average of max(p_reward) 
    bandit.maximum_reward_rate = np.mean(np.sum(bandit.p_reward, axis = 0)) #??? Method 2: Average of sum(p_reward).   [Corrado et al 2005: efficienty = 50% for choosing only one color]
    # bandit.maximum_reward_rate = np.sum(np.any(bandit.reward_available, axis = 0)) / bandit.n_trials  #??? Method 3: Maximum reward given the fixed reward_available (one choice per trial constraint) [Sugrue 2004???]
    # bandit.maximum_reward_rate = np.sum(np.sum(bandit.reward_available, axis = 0)) / bandit.n_trials  #??? Method 4: Sum of all ever-baited rewards (not fair)  

    bandit.foraging_efficiency = bandit.actual_reward_rate / bandit.maximum_reward_rate
    
    # -- 2. Blockwise statistics --
    temp_nans = np.zeros(bandit.n_blocks)
    temp_nans[:] = np.nan   # Better way?
    
    bandit.blockwise_choice_fraction = temp_nans.copy()
    bandit.blockwise_reward_fraction = temp_nans.copy()
    bandit.blockwise_log_choice_ratio = temp_nans.copy()
    bandit.blockwise_log_reward_ratio = temp_nans.copy()
    
    bandit.block_trans_time = np.cumsum(np.hstack([0,bandit.n_trials_per_block]))
    
    for i_block in range(bandit.n_blocks):   # For each block in this session
        trial_range = np.r_[bandit.block_trans_time[i_block] : bandit.block_trans_time[i_block+1]]  # r_ trick
       
        choice_R = np.sum(bandit.choice_history[trial_range] == RIGHT)
        choice_L = np.sum(bandit.choice_history[trial_range] == LEFT)
        rew_R = np.sum(bandit.reward_history[RIGHT, trial_range])
        rew_L = np.sum(bandit.reward_history[LEFT, trial_range])
                
        if (rew_R + rew_L):    # Non-zero total reward. Otherwise, leaves nan
            bandit.blockwise_choice_fraction[i_block] = choice_R / (choice_R + choice_L)
            bandit.blockwise_reward_fraction[i_block] = rew_R / (rew_R + rew_L)
        
        if all([rew_R, rew_L, choice_R, choice_L]):   # All non-zero. Otherwise, leaves nan
            bandit.blockwise_log_choice_ratio[i_block] = np.log(choice_R / choice_L)
            bandit.blockwise_log_reward_ratio[i_block] = np.log(rew_R / rew_L)
        
    return bandit   # For apply_async, in-place change is impossible since each worker uses "bandit" as 
                    # an independent local object. So I have to return "bandit" explicitly

def run_sessions_parallel(bandit, n_sessions = global_n_sessions):  
# =============================================================================
#  Run simulations with the SAME bandit in serial or in parallel, repeating n_sessions
# =============================================================================
       
   
    # Generate a series of deepcopys of bandit to make them independent!!
    bandits_all_sessions = []
    [bandits_all_sessions.append(copy.deepcopy(bandit)) for ss in range(n_sessions)]
        
    if 'serial' in methods:  # Serial computing (for test)
        start = time.time()         
        for ss, bb in tqdm(enumerate(bandits_all_sessions), total = n_sessions, desc='serial'):     # trange: progress bar. HH
            run_one_session(bb)     # There is no need to assign back the resulting bandit. (Modified inside run_one_session())
             
        print('--- serial finished in %g s ---\n' % (time.time()-start))
        

    if 'apply_async' in methods:    # Parallel computing using multiprocessing.apply_async()
        start = time.time()
        
        # Note the "," in (bb,). See here https://stackoverflow.com/questions/29585910/why-is-multiprocessings-apply-async-so-picky
        result_ids = [pool.apply_async(run_one_session, args = (bb,)) for bb in bandits_all_sessions]  
                        
        for ss, result_id in tqdm(enumerate(result_ids), total = n_sessions, desc='apply_async'):
            # For apply_async, the assignment is required, because the bb passed to the workers are local independent copys.
            bandits_all_sessions[ss] = result_id.get()  
            
        print('\n--- apply_async finished in %g s--- \n' % (time.time()-start), flush=True)
        
        
    plot_all_sessions(bandits_all_sessions)  # Plot summary statistics over repeated sessions for this bandit
   
    
    return 


def figure_2_2():
    
    title_txt = '\n=== Figure 2.2: Sample-average \nDifferent eps (%g runs)===\n' % global_n_sessions
    print(title_txt, flush = True)
        
    epsilons = [0.1]
    effective_tau = 5
    step_size = 1 - np.exp(-1/effective_tau)
    
    # Generate a series of Bandit objects using different eps. HH
    
    bandit = [Bandit(epsilon = eps, step_size = step_size, if_baited = True) for eps in epsilons]   # Use the [f(xxx) for xxx in yyy] trick. HH!!!
    
    # Run simulations, return best_action_counts and rewards. HH
    run_sessions_parallel(bandit[0])
    
# =============================================================================
# 
#     plt.figure(figsize=(10, 20))
#     plt.clf
#     plt.subplot(2, 1, 1)
#     
#     # Plotting average rewards. Use zip(epsilons, rewards), and plt.plot(rewards, label = 'xxx %X %X' %(X,X))
#     
#     
#     for eps, rew in zip(epsilons, rewards):
#         h = plt.plot(rew[0], label = 'epsilon = %2g' %eps)
#         plt.fill_between(np.arange(0, n_trials), rew[0] - rew[1], rew[0] + rew[1], alpha = 0.2, color = h[0].get_color())
#     
#     
#     plt.xlabel('steps')
#     plt.ylabel('average reward')
#     plt.legend()
# 
#     plt.subplot(2, 1, 2)
#     
# 
#     plt.savefig('one_session.png')
# #    plt.close()
# =============================================================================


if __name__ == '__main__':
    
    n_worker = mp.cpu_count()
    methods = [ 
                'serial',
                # 'apply_async'   # This is best till now!!!
              ]
    
    if any([x in methods for x in ('apply_async','map','imap_unordered','imap')]):
        pool = mp.Pool(processes = n_worker)

    figure_2_2()
    
    if any([x in methods for x in ('apply_async','map','imap_unordered','imap')]):
        pool.close()   # This is a good practice
