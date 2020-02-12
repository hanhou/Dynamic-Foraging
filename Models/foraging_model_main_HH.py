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
global_n_runs = 2000

def one_run(bandit):     
# =============================================================================
# Make one-run independently
# =============================================================================
    
    # === Perform one run ===
    bandit.reset()
    for t in range(bandit.n_trials):        
        # Loop: (Act --> Reward & New state)
        action = bandit.act()
        bandit.step(action)
        
    # === Summarize results for this run ===
    results_this_run = dict()
    results_this_run['choice_history'] = bandit.choice_history
    results_this_run['reward_history'] = bandit.reward_history
      
    return results_this_run


def repeat_runs(bandit, n_runs = global_n_runs):  
# =============================================================================
#  Run simulations in serial or in parallel, for the SAME bandit, repeating n_runs
# =============================================================================
       
    choice_all_runs = np.zeros([n_runs, bandit.n_trials])
    reward_all_runs = np.zeros([n_runs, global_k_arm, bandit.n_trials])  # Assuming all bandits have the same n_trials
    
    if 'serial' in methods:
    
        start = time.time()         # Serial computing
        for r in tqdm(range(n_runs), desc='serial'):     # trange: progress bar. HH
            results_this_run = one_run(bandit)   
            choice_all_runs[r, :] = results_this_run['choice_history']
            reward_all_runs[r, :, :] = results_this_run['reward_history']
            
        print('--- serial finished in %g s ---\n' % (time.time()-start))

    if 'apply_async' in methods:    # Using multiprocessing.apply_async()

        start = time.time()
       
        result_ids = [pool.apply_async(one_run, args=(bandit)) for r in range(n_runs)]
                        
        for r, result_id in tqdm(enumerate(result_ids), total = n_runs, desc='apply_async'):
            results_this_run = one_run(bandit)
            choice_all_runs[r, :] = results_this_run['choice_history']
            reward_all_runs[r, :, :] = results_this_run['reward_history']
            
        print('\n--- apply_async finished in %g s--- \n' % (time.time()-start), flush=True)


    plot_one_session(bandit)  # This is actually the "last run" of all the repeated runs

    
    return 


def figure_2_2(runs = 1, n_trials = global_n_trials):
    
    title_txt = '\n=== Figure 2.2: Sample-average \nDifferent eps (%g runs)===\n' % runs
    print(title_txt, flush = True)
        
    epsilons = [0.1]
    effective_tau = 5
    step_size = 1 - np.exp(-1/effective_tau)
    
    # Generate a series of Bandit objects using different eps. HH
    
    bandit = [Bandit(epsilon = eps, step_size = step_size, if_baited = True, n_trials = global_n_trials) for eps in epsilons]   # Use the [f(xxx) for xxx in yyy] trick. HH!!!
    
    # Run simulations, return best_action_counts and rewards. HH
    repeat_runs(bandit[0], runs)
    
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
    methods = [ 'serial',
                # 'apply_async'   # This is best till now!!!
              ]
    
    if any([x in methods for x in ('apply_async','map','imap_unordered','imap')]):
        pool = mp.Pool(processes = n_worker)

    figure_2_2()
