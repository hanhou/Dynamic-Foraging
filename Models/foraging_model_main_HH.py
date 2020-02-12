# =============================================================================
# Main function for simulating foraging_model_HH
# =============================================================================
"""
Created on Wed Feb 12 14:11:12 2020

@author: Han
"""

import numpy as np
from tqdm import tqdm  # For progress bar. HH
import time
import multiprocessing as mp

# Import my own modules
from foraging_model_HH import Bandit
from foraging_model_plots_HH import plot_one_session

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
    # - Blockwise statistics -
    
    results_this_session = dict()
    results_this_session['n_trials'] = bandit.n_trials
    results_this_session['p_reward'] = bandit.p_reward
    results_this_session['reward_available'] = bandit.reward_available
    results_this_session['choice_history'] = bandit.choice_history
    results_this_session['reward_history'] = bandit.reward_history
    results_this_session['description'] = bandit.description
      
    return results_this_session


def run_sessions_parallel(bandit, n_sessions = global_n_sessions):  
# =============================================================================
#  Run simulations with the SAME bandit in serial or in parallel, repeating n_sessions
# =============================================================================
       
    choice_all_runs = np.zeros([n_sessions, bandit.n_trials])
    reward_all_runs = np.zeros([n_sessions, global_k_arm, bandit.n_trials])  # Assuming all bandits have the same n_trials
    
    
    # bandits_all_sessions = [bandit] * n_sessions  # This does not work because all bandits have the same REFERENCE!
    results_all_sessions = []
    
    if 'serial' in methods:
    
        start = time.time()         # Serial computing
        for ss in tqdm(range(n_sessions), total = n_sessions, desc='serial'):     # trange: progress bar. HH
            results_all_sessions.append(run_one_session(bandit))     # Add bandit to results
            choice_all_runs[ss, :] = results_all_sessions[ss]['choice_history']
            reward_all_runs[ss, :, :] = results_all_sessions[ss]['reward_history']
            
        print('--- serial finished in %g s ---\n' % (time.time()-start))

    if 'apply_async' in methods:    # Using multiprocessing.apply_async()

        start = time.time()
        
        # Note the "," in (bandit,). See here https://stackoverflow.com/questions/29585910/why-is-multiprocessings-apply-async-so-picky
        result_ids = [pool.apply_async(run_one_session, args = (bandit,)) for ss in range(n_sessions)]  
                        
        for ss, result_id in tqdm(enumerate(result_ids), total = n_sessions, desc='apply_async'):
            results_all_sessions.append(result_id.get())
            choice_all_runs[ss, :] = results_all_sessions[ss]['choice_history']
            reward_all_runs[ss, :, :] = results_all_sessions[ss]['reward_history']
            
        print('\n--- apply_async finished in %g s--- \n' % (time.time()-start), flush=True)

    plot_one_session(results_all_sessions[0])  # Plot example session
   
    
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
                # 'serial',
                'apply_async'   # This is best till now!!!
              ]
    
    if any([x in methods for x in ('apply_async','map','imap_unordered','imap')]):
        pool = mp.Pool(processes = n_worker)

    figure_2_2()
    
    if any([x in methods for x in ('apply_async','map','imap_unordered','imap')]):
        pool.close()   # This is a good practice
