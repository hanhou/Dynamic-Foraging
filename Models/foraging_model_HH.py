# =============================================================================
# Models for dynamic foraging task
# 
# - Use apply_async() in multiprocessing for parallel computing
# - Modified from the code for Sutton & Barto's RL book, Chapter 02
#   1. 2-bandit problem
#   2. Introducing "baiting" property
#   3. Baiting dynamics reflects in the reward probability rather than reward amplitude
# 
# 
# Han Hou @ Houston Feb 2020
# Svoboda lab
# =============================================================================

#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Artem Oboturov(oboturov@gmail.com)                             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})

import numpy as np
from tqdm import tqdm  # For progress bar. HH
    
import time
import multiprocessing as mp
    
#matplotlib.use('Agg')  # Agg -> non-GUI backend. HH
# matplotlib.use('qt5agg')  # We can see the figure by qt5. HH

LEFT = 0
RIGHT = 1

global_k_arm = 2
global_n_trials = 500  # To cope with the one-argument limitation of map/imap
global_n_trials_per_block_base = 80
global_n_trials_per_block_sd = 20

global_n_runs = 2000

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class Bandit:
    
    # =============================================================================
    # Different foragers
    #  1.'Sutton_Barto': Track a 'return' R (averaged over 'chosen trials' not all trials), constant step, epsilon-greedy      
    # =============================================================================
    # @k_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations
    
    def __init__(self, k_arm = global_k_arm, n_trials = global_n_trials, epsilon = 0.1, step_size = 0.1, 
                 if_baited = True, forager = 'Sutton_Barto'):

        self.k = k_arm
        self.n_trials = n_trials
        self.step_size = step_size
        self.epsilon = epsilon
        self.forager = forager
        self.if_baited = if_baited
        
        self.description = '%s, epsi = %g, alpha = %g' % (forager,epsilon,step_size)
  

    def reset(self):
        
        # Initialization
        self.time = 0
        self.q_estimation = np.zeros(self.k) # Estimation for each action 
        self.choice_history = np.zeros(self.n_trials)  # Choice history
        self.reward_history = np.zeros([self.k, self.n_trials])    # Reward history, separated for each port (Corrado Newsome 2005)
        
        # Generate baiting prob in block structure
        [self.p_reward, self.n_trials_per_block] = self.generate_p_reward()
        
        # Prepare reward for the first trial
        # For example, [0,1] represents there is reward baited at the RIGHT but not LEFT port.
        self.reward_status = np.zeros([self.k, self.n_trials])    # Reward history, separated for each port (Corrado Newsome 2005)
        self.reward_status[:,0] = (np.random.uniform(0,1,self.k) < self.p_reward[:, self.time]).astype(int)

    # =============================================================================
    #  Generate p_reward array (Bari-Cohen 2019)
    # =============================================================================
    def generate_p_reward(self, n_trials_per_block_base = global_n_trials_per_block_base, n_trials_per_block_sd = global_n_trials_per_block_sd,
                          reward_ratio_pairs = [[.4,.05],[.3857,.0643],[.3375,.1125],[.225,.225]]):
        
        n_trials_now = 0
        n_trials_per_block = []  
        p_reward = np.zeros([2,self.n_trials])
        
        # Fill in trials until the required length
        while n_trials_now < self.n_trials:
        
            # Number of trials in each block (Gaussian distribution)
            # I treat p_reward[0,1] as the ENTIRE lists of reward probability. RIGHT = 0, LEFT = 1. HH
            n_trials_this_block = np.rint(np.random.normal(0, n_trials_per_block_sd) + n_trials_per_block_base).astype(int) 
            n_trials_this_block = min(n_trials_this_block, self.n_trials - n_trials_now)
            
            n_trials_per_block.append(n_trials_this_block)
                  
            # Get values to fill for this block
            if n_trials_now == 0:  # The first block is set to 50% reward rate (as Marton did)
                p_reward_this_block = np.array([[sum(reward_ratio_pairs[0])/2] * 2])  # Note the outer brackets
            else:
                # Choose reward_ratio_pair
                if not(np.diff(p_reward_this_block)):   # If we had equal p_reward in the last trial
                    ratiopairidx = np.random.choice(range(len(reward_ratio_pairs)-1))   # We should not let it happen again immediately
                else:
                    ratiopairidx = np.random.choice(range(len(reward_ratio_pairs)))
                    
                p_reward_this_block = np.array([reward_ratio_pairs[ratiopairidx]])   # Note the outer brackets

                # To ensure flipping of p_reward during transition (Marton)
                if len(n_trials_per_block) % 2:     
                    p_reward_this_block = np.flip(p_reward_this_block)
                
            # Fill in trials for this block
            p_reward[:, n_trials_now : n_trials_now + n_trials_this_block] = p_reward_this_block.T
            n_trials_now += n_trials_this_block


        return p_reward, np.array(n_trials_per_block)


    def act(self):
    # =============================================================================
    #   Make a choice for this trial
    # =============================================================================
        
        if self.forager == 'Sutton_Barto':  # Estimation of "local return" + epsi-greedy
            if np.random.rand() < self.epsilon:  # Forced exploration with the prob. of epsilon
                choice = np.random.choice(self.k)
            else:    # Else, do hardmax (greedy)
                choice = np.random.choice(np.where(self.q_estimation == self.q_estimation.max())[0])
                
        self.choice_history[self.time] = choice
        
        return choice
            
  
    def step(self, choice):
    # =============================================================================
    #   Get feedback from the environment
    #   Update value/preference estimation
    # =============================================================================
        
        # -- Generate reward and make the state transition (i.e., prepare reward for the next trial) --
        reward = self.reward_status[choice, self.time]    
        self.reward_history[choice, self.time] = reward   # Note that according to Sutton & Barto's convention,
                                                          # this update should belong to time t+1.
        
        reward_status_after_choice = self.reward_status[:, self.time].copy()  # An intermediate reward status. Note the .copy()!
        reward_status_after_choice [choice] = 0   # The reward is depleted at the chosen lick port.
        
        self.time += 1   # Time ticks here.
        if self.time == global_n_trials: return;   # Session terminates
        
        # For the next reward status, the "or" statement ensures the baiting property.
        self.reward_status[:, self.time] = np.logical_or(  reward_status_after_choice * self.if_baited,    
                                           np.random.uniform(0,1,self.k) < self.p_reward[:,self.time]).astype(int)  
            
        # -- Update value estimation --
        if self.forager == 'Sutton_Barto':
            # Update estimation with constant step size
            self.q_estimation[choice] += (reward - self.q_estimation[choice]) * self.step_size            
            
        return reward
  
 
def one_run(bandit, if_plot=False):   # I put "bandit" at the first place because map/imap_xxx only receive one argument.    
# =============================================================================
# Make one-run independently
# =============================================================================
    
    # Perform one run
    bandit.reset()
    
    for t in range(bandit.n_trials):        
        # Act --> Reward & New state
        action = bandit.act()
        bandit.step(action)
        
    choice_this_run = bandit.choice_history
    rewards_this_run = bandit.reward_history
      
    if if_plot:
        plot_one_session(bandit)  # All histories have been saved in the bandit object
        
    return choice_this_run, rewards_this_run


def repeat_runs(bandit, n_runs = global_n_runs):  
# =============================================================================
#  Run simulations in serial or in parallel, for the SAME bandit, repeating n_runs
# =============================================================================
       
    choice_all_runs = np.zeros([n_runs, bandit.n_trials])
    reward_all_runs = np.zeros([n_runs, global_k_arm, bandit.n_trials])  # Assuming all bandits have the same n_trials
    
    if 'serial' in methods:
    
        start = time.time()         # Serial computing
        for r in tqdm(range(n_runs), desc='serial'):     # trange: progress bar. HH
            choice_all_runs[r, :], reward_all_runs[r, :, :] = one_run(bandit, if_plot = (r==0))   # Plot the first session
            
        print('--- serial finished in %g s ---\n' % (time.time()-start))

    if 'apply_async' in methods:    # Using multiprocessing.apply_async()

        start = time.time()
       
        result_ids = [pool.apply_async(one_run, args=(bandit, r==0)) for r in range(n_runs)]
                        
        for r, result_id in tqdm(enumerate(result_ids), total = n_runs, desc='apply_async'):
            choice_all_runs[r, :], reward_all_runs[r, :, :] = result_id.get()  # Put it in results
            
        print('\n--- apply_async finished in %g s--- \n' % (time.time()-start), flush=True)

      
    return 

def plot_one_session(bandit, plottype='2lickport'):     # Part of code from Marton
    
    # Fetch data
    choice_history = bandit.choice_history
    reward_history = bandit.reward_history
    rewarded_trials = np.any(reward_history, axis = 0)
    unrewarded_trials = np.logical_not(rewarded_trials)
    
    # Foraging efficiency = Sum of actual rewards / Maximum number of rewards that could have been collected
    actual_reward_rate = np.sum(reward_history) / bandit.n_trials
    
    '''Don't know which one is better'''
    # maximum_reward_rate = np.mean(np.max(bandit.p_reward, axis = 0)) #??? Method 1: Average of max(p_reward) 
    maximum_reward_rate = np.mean(np.sum(bandit.p_reward, axis = 0)) #??? Method 2: Average of sum(p_reward).   [Corrado et al 2005: efficienty = 50% for choosing only one color]
    # maximum_reward_rate = np.sum(np.any(bandit.reward_status, axis = 0)) / bandit.n_trials  #??? Method 3: Maximum reward given the fixed reward_status (one choice per trial constraint) [Sugrue 2004???]
    # maximum_reward_rate = np.sum(np.sum(bandit.reward_status, axis = 0)) / bandit.n_trials  #??? Method 4: Sum of all ever-baited rewards (not fair)  

    foraging_efficiency = actual_reward_rate / maximum_reward_rate
    
    p_reward_ratio = bandit.p_reward[RIGHT,:] / (np.sum(bandit.p_reward, axis = 0))
    
    fig = plt.figure()
    ax1 = plt.subplot(2,1,1)

    # Rewarded trials
    ax1.plot(np.nonzero(rewarded_trials)[0], choice_history[rewarded_trials], 'k|',color='black',markersize=30, markeredgewidth=2)
    
    # Unrewarded trials
    ax1.plot(np.nonzero(unrewarded_trials)[0], choice_history[unrewarded_trials], '|',color='gray', markersize=15, markeredgewidth=1)
    
    # Baited probability and smoothed choice history
    ax1.plot(np.arange(0, bandit.n_trials), p_reward_ratio, color='orange')
    ax1.plot(moving_average(choice_history, 10) , color='black')
    
    ax1.set_yticks([0,1])
    ax1.set_yticklabels(['Left','Right'])
    plt.xlabel('choice #')
    
    # Reward rate
    plt.title('%s, efficiency = %.02f' % (bandit.description, foraging_efficiency), fontsize = 10)
    fig.show()
  

    ax1.set_title(wr_name + '   -   session: ' + str(session) + ' - '+str(df_session['session_date'][0]))
    ax1.legend(fontsize='small',loc = 'upper right')
    
    ax2=fig.add_axes([0,-1,2,.8])
    ax2.plot(df_behaviortrial['trial'],df_behaviortrial['p_reward_left'],'r-')
    ax2.plot(df_behaviortrial['trial'],df_behaviortrial['p_reward_right'],'b-')
    if plottype == '3lickport':
        ax2.plot(df_behaviortrial['trial'],df_behaviortrial['p_reward_middle'],'g-')
    ax2.set_ylabel('Reward probability')
    ax2.set_xlabel('Trial #')
    if plottype == '3lickport':
        legenda = ['left','right','middle']
    else:
        legenda = ['left','right']
    ax2.legend(legenda,fontsize='small',loc = 'upper right')
    

def figure_2_2(runs = 1, n_trials = global_n_trials):
    
    title_txt = '\n=== Figure 2.2: Sample-average \nDifferent eps (%g runs)===\n' % runs
    print(title_txt, flush = True)
        
    epsilons = [.3]
    effective_tau = 5
    step_size = 1 - np.exp(-1/effective_tau)
    
    # Generate a series of Bandit objects using different eps. HH
    
    bandit = [Bandit(epsilon = eps, step_size = step_size, if_baited = True) for eps in epsilons]   # Use the [f(xxx) for xxx in yyy] trick. HH!!!
    
    # Run simulations, return best_action_counts and rewards. HH
    repeat_runs(bandit[0], runs)
    

    plt.figure(figsize=(10, 20))
    plt.clf
    plt.subplot(2, 1, 1)
    
    # Plotting average rewards. Use zip(epsilons, rewards), and plt.plot(rewards, label = 'xxx %X %X' %(X,X))
    
    
    for eps, rew in zip(epsilons, rewards):
        h = plt.plot(rew[0], label = 'epsilon = %2g' %eps)
        plt.fill_between(np.arange(0, n_trials), rew[0] - rew[1], rew[0] + rew[1], alpha = 0.2, color = h[0].get_color())
    
    
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    

    plt.savefig('one_session.png')
#    plt.close()


if __name__ == '__main__':
    
    n_worker = mp.cpu_count()
    methods = [ 'serial',
                # 'apply_async'   # This is best till now!!!
              ]
    
    if any([x in methods for x in ('apply_async','map','imap_unordered','imap')]):
        pool = mp.Pool(processes = n_worker)

    figure_2_2()
