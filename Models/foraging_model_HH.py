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


import numpy as np

global_n_trials_per_block_base = 80
global_n_trials_per_block_sd = 20


class Bandit:
    
    # =============================================================================
    # Different foragers
    #  1.'Sutton_Barto': Track a 'return' R (averaged over 'chosen trials' not all trials), constant step, epsilon-greedy      
    # =============================================================================
    # @k_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations
    
    def __init__(self, k_arm = 2, n_trials = 1000, epsilon = 0.1, step_size = 0.1, 
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
        self.reward_available = np.zeros([self.k, self.n_trials])    # Reward history, separated for each port (Corrado Newsome 2005)
        self.reward_available[:,0] = (np.random.uniform(0,1,self.k) < self.p_reward[:, self.time]).astype(int)

    # =============================================================================
    #  Generate p_reward array (Bari-Cohen 2019)
    # =============================================================================
    def generate_p_reward(self, n_trials_per_block_base = global_n_trials_per_block_base, 
                                n_trials_per_block_sd = global_n_trials_per_block_sd,
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
        reward = self.reward_available[choice, self.time]    
        self.reward_history[choice, self.time] = reward   # Note that according to Sutton & Barto's convention,
                                                          # this update should belong to time t+1.
        
        reward_available_after_choice = self.reward_available[:, self.time].copy()  # An intermediate reward status. Note the .copy()!
        reward_available_after_choice [choice] = 0   # The reward is depleted at the chosen lick port.
        
        self.time += 1   # Time ticks here.
        if self.time == self.n_trials: return;   # Session terminates
        
        # For the next reward status, the "or" statement ensures the baiting property, gated by self.if_baited.
        self.reward_available[:, self.time] = np.logical_or(  reward_available_after_choice * self.if_baited,    
                                           np.random.uniform(0,1,self.k) < self.p_reward[:,self.time]).astype(int)  
            
        # -- Update value estimation --
        if self.forager == 'Sutton_Barto':
            # Update estimation with constant step size
            self.q_estimation[choice] += (reward - self.q_estimation[choice]) * self.step_size            
            
        return reward
  
 
