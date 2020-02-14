# =============================================================================
# Models for dynamic foraging task
# =============================================================================
# - Initially modified from the code for Sutton & Barto's RL book, Chapter 02
#   1. 10-bandit --> 2-bandit problem
#   2. Introducing "baiting" property
#   3. Baiting dynamics reflects in the reward probability rather than reward amplitude
#
# - Forager types:
#   1. 'Random'
#   2. 'AlwaysLEFT': only chooses LEFT
#   3. 'IdealGreedy': knows p_reward and always chooses the largest one
#   4. 'Sutton_Barto':   return  ->   exp filter                                    -> epsilon-greedy
#   5. 'Sugrue2004':     income  ->   exp filter   ->  fractional                   -> epsilon-Poisson 
#   5.1 'IIgaya2019':    income  ->  2-exp filter  ->  fractional                   -> epsilon-Poisson (epsilon has the same effect as tau_long??)
#   6. 'Corrado2005':    income  ->  2-exp filter  ->  softmax ( = diff + sigmoid)  -> epsilon-Poisson (epsilon has the same effect as tau_long??)
# 
# Han Hou @ Houston Feb 2020
# Svoboda & Li lab
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

LEFT = 0
RIGHT = 1

global_block_size_mean = 100
global_block_size_sd = 30

softmax = lambda x, softmax_temperature: np.exp(x/softmax_temperature)/np.sum(np.exp(x/softmax_temperature))  # Accept np.arrays


class Bandit:
    
    # @k_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations
    
    def __init__(self, k_arm = 2, n_trials = 1000, if_baited = True, epsilon = 0, softmax_temperature = np.nan, random_before_total_reward = 0, # Shared paras
                 forager = 'Sutton_Barto', 
                 step_size = 0.1,                        # For 'Sutton_Barto'. Other paras: epsilon
                 tau = 20,                               # For 'Sugrue2004'  . Other paras: epsilon
                 tau_fast = 2, tau_slow = 15, w_tau_slow = 0.3    # For 'Corrado2005' or 'Iigaya2019'. Other paras: epsilon, softmax_temperature 
                 ):     

        self.k = k_arm
        self.n_trials = n_trials
        self.if_baited = if_baited
        self.forager = forager
        
        self.epsilon = epsilon
        self.step_size = step_size
        self.softmax_temperature = softmax_temperature
        self.random_before_total_reward = random_before_total_reward
        
        self.tau = tau
        self.tau_fast = tau_fast
        self.tau_slow  = tau_slow
        self.w_tau_slow = w_tau_slow
          
    def reset(self):
        
        #  print(self)
        
        # Initialization
        self.time = 0
        self.choice_history = np.zeros([1,self.n_trials])  # Choice history
        self.q_estimation = np.zeros([self.k, self.n_trials]) # Estimation for each action (e.g., Q in Bari2019, L-stage scalar value in Corrado2005) 
        self.reward_history = np.zeros([self.k, self.n_trials])    # Reward history, separated for each port (Corrado Newsome 2005)
        
        # Generate baiting prob in block structure
        self.generate_p_reward()
        
        # Prepare reward for the first trial
        # For example, [0,1] represents there is reward baited at the RIGHT but not LEFT port.
        self.reward_available = np.zeros([self.k, self.n_trials])    # Reward history, separated for each port (Corrado Newsome 2005)
        self.reward_available[:,0] = (np.random.uniform(0,1,self.k) < self.p_reward[:, self.time]).astype(int)
        
        # Forager-specific
        if self.forager == 'Sutton_Barto':
            self.description = 'Sutton_Barto, epsi = %g, alpha = %g' % (self.epsilon, self.step_size)
            
        elif self.forager == 'Sugrue2004':
            self.description = 'Sugrue2004, tau = %g, epsi = %g' % (self.tau, self.epsilon)
            
            reversed_t = np.flipud(np.arange(self.n_trials))  # Use the full length of the session just in case of an extremely large tau.
            self.history_filter = np.exp(-reversed_t / self.tau)
            self.q_estimation[:] = 1/self.k   # To be strict
            
        elif self.forager in ['Corrado2005', 'IIgaya2019']:
            self.description = '%s, tau_fast = %g, tau_slow = %g, w_tau_slow = %g, softmax_temp = %g, epsi = %g, random_before_total_reward = %g' % \
                                            (self.forager, self.tau_fast, self.tau_slow, self.w_tau_slow , self.softmax_temperature, self.epsilon, self.random_before_total_reward)
                                            
            reversed_t = np.flipud(np.arange(self.n_trials))  # Use the full length of the session just in case of an extremely large tau.
            self.history_filter = (1-self.w_tau_slow) * np.exp(-reversed_t / self.tau_fast) + self.w_tau_slow * np.exp(-reversed_t / self.tau_slow)
            self.q_estimation[:] = 1/self.k   # To be strict
                  
        else:
            self.description = self.forager

        

    def generate_p_reward(self, block_size_base = global_block_size_mean, 
                                block_size_sd = global_block_size_sd,
                                p_reward_pairs = [[.4,.05],[.3857,.0643],[.3375,.1125],[.225,.225]]):  # (Bari-Cohen 2019)
        
        n_trials_now = 0
        block_size = []  
        p_reward = np.zeros([2,self.n_trials])
        
        # Fill in trials until the required length
        while n_trials_now < self.n_trials:
        
            # Number of trials in each block (Gaussian distribution)
            # I treat p_reward[0,1] as the ENTIRE lists of reward probability. RIGHT = 0, LEFT = 1. HH
            n_trials_this_block = np.rint(np.random.normal(0, block_size_sd) + block_size_base).astype(int) 
            n_trials_this_block = min(n_trials_this_block, self.n_trials - n_trials_now)
            
            block_size.append(n_trials_this_block)
                              
            # Get values to fill for this block
            if n_trials_now == -1:  # If 0, the first block is set to 50% reward rate (as Marton did)
                p_reward_this_block = np.array([[sum(p_reward_pairs[0])/2] * 2])  # Note the outer brackets
            else:
                # Choose reward_ratio_pair
                if n_trials_now > 0 and not(np.diff(p_reward_this_block)):   # If we had equal p_reward in the last block
                    pair_idx = np.random.choice(range(len(p_reward_pairs)-1))   # We should not let it happen again immediately
                else:
                    pair_idx = np.random.choice(range(len(p_reward_pairs)))
                    
                p_reward_this_block = np.array([p_reward_pairs[pair_idx]])   # Note the outer brackets

                # To ensure flipping of p_reward during transition (Marton)
                if len(block_size) % 2:     
                    p_reward_this_block = np.flip(p_reward_this_block)
                
            # Fill in trials for this block
            p_reward[:, n_trials_now : n_trials_now + n_trials_this_block] = p_reward_this_block.T
            n_trials_now += n_trials_this_block
        
        self.n_blocks = len(block_size)
        self.p_reward = p_reward
        self.block_size  = np.array(block_size)
        self.p_reward_fraction = p_reward[RIGHT,:] / (np.sum(p_reward, axis = 0))   # For future use
        self.p_reward_ratio = p_reward[RIGHT,:] / p_reward[LEFT,:]   # For future use


    def act(self):
        # =============================================================================
        #   Make a choice for this trial
        # =============================================================================
        # - Forager types:
        #   1. 'Random'
        #   2. 'AlwaysLEFT': only chooses LEFT
        #   3. 'IdealGreedy': knows p_reward and always chooses the largest one
        #   4. 'Sutton_Barto':   return  ->   exp filter                                    -> epsilon-greedy
        #   5. 'Sugrue2004':     income  ->   exp filter   ->  fractional                   -> epsilon-Poisson 
        #   5.1 'IIgaya2019':    income  ->  2-exp filter  ->  fractional                   -> epsilon-Poisson (epsilon has the same effect as tau_long??)
        #   6. 'Corrado2005':    income  ->  2-exp filter  ->  softmax ( = diff + sigmoid)  -> epsilon-Poisson (epsilon has the same effect as tau_long??)
               
        if self.forager == 'Random': 
            choice = np.random.choice(self.k)
            
        elif self.forager == 'AlwaysLEFT':
            choice = LEFT
            
        elif self.forager == 'IdealGreedy':
            choice = np.random.choice(np.where(self.p_reward[:,self.time] == self.p_reward[:,self.time].max())[0])
                
        else:
            if np.random.rand() < self.epsilon or np.sum(self.reward_history) < self.random_before_total_reward: 
                # Forced exploration with the prob. of epsilon (to avoid AlwaysLEFT/RIGHT in Sugrue2004...) or before some rewards are collected #???
                choice = np.random.choice(self.k)
                
            else:   # Forager-dependent
                if self.forager == 'Sutton_Barto':   # Greedy
                    choice = np.random.choice(np.where(self.q_estimation[:, self.time] == self.q_estimation[:, self.time].max())[0])
                    
                elif self.forager in ['Sugrue2004', 'Corrado2005', 'IIgaya2019']:   # Poisson
                    if np.random.rand() < self.q_estimation[LEFT, self.time]:
                        choice = LEFT
                    else:
                        choice = RIGHT
                
        self.choice_history[0, self.time] = choice
        
        return choice
            
  
    def step(self, choice):
           
        # =============================================================================
        #  Generate reward and make the state transition (i.e., prepare reward for the next trial) --
        # =============================================================================
        reward = self.reward_available[choice, self.time]    
        self.reward_history[choice, self.time] = reward   # Note that according to Sutton & Barto's convention,
                                                          # this update should belong to time t+1, but here I use t for simplicity.
        
        reward_available_after_choice = self.reward_available[:, self.time].copy()  # An intermediate reward status. Note the .copy()!
        reward_available_after_choice [choice] = 0   # The reward is depleted at the chosen lick port.
        
        self.time += 1   # Time ticks here.
        if self.time == self.n_trials: 
            return;   # Session terminates
        
        # For the next reward status, the "or" statement ensures the baiting property, gated by self.if_baited.
        self.reward_available[:, self.time] = np.logical_or(  reward_available_after_choice * self.if_baited,    
                                           np.random.uniform(0,1,self.k) < self.p_reward[:,self.time]).astype(int)  
            
        # =============================================================================
        #  Update value estimation (or Poisson choice probability)
        # =============================================================================
        # - Forager types:
        #   1. 'Random'
        #   2. 'AlwaysLEFT': only chooses LEFT
        #   3. 'IdealGreedy': knows p_reward and always chooses the largest one
        #   4. 'Sutton_Barto':   return  ->   exp filter                                    -> epsilon-greedy
        #   5. 'Sugrue2004':     income  ->   exp filter   ->  fractional                   -> epsilon-Poisson 
        #   5.1 'IIgaya2019':    income  ->  2-exp filter  ->  fractional                   -> epsilon-Poisson (epsilon has the same effect as tau_long??)
        #   6. 'Corrado2005':    income  ->  2-exp filter  ->  softmax ( = diff + sigmoid)  -> epsilon-Poisson (epsilon has the same effect as tau_long??)
                
        if self.forager == 'Sutton_Barto':    
            # Local ~return~
            # Note: It's "return" rather than "income" because only the ~chosen~ one is updated here
            self.q_estimation[:, self.time] = self.q_estimation[:, self.time - 1]  # Don't forget cache the old values!
            self.q_estimation[choice, self.time] += (reward - self.q_estimation[choice, self.time]) * self.step_size   
            
        elif self.forager in ['Sugrue2004', 'IIgaya2019']:
            # Fractional local income
            # Note: It's "income" because the following computations do not dependent on the current ~choice~.
            
            # 1. Local income = Reward history + exp filter in Sugrue or 2-exp filter in IIgaya
            valid_reward_history = self.reward_history[:, :self.time]   # History till now
            valid_filter = self.history_filter[-self.time:]    # Corresponding filter
            local_income = np.sum(valid_reward_history * valid_filter, axis = 1)
            
            # 2. Poisson choice probability = Fractional local income
            if np.sum(local_income) == 0:
                # 50%-to-50%
                self.q_estimation[:, self.time] = [1/self.k] * self.k
            else:
                # Local fractional income
                self.q_estimation[:, self.time] = local_income / np.sum(local_income)
                
        elif self.forager == 'Corrado2005':
            # Softmaxed local income
            
            # 1. Local income = Reward history + hyperbolic (2-exps) filter
            valid_reward_history = self.reward_history[:, :self.time]   # History till now
            valid_filter = self.history_filter[-self.time:]    # Corresponding filter
            local_income = np.sum(valid_reward_history * valid_filter, axis = 1)
            
            # 2. Poisson choice probability = Softmaxed local income (Note: Equivalent to "difference + sigmoid" in [Corrado etal 2005], for 2lp case)
            self.q_estimation[:, self.time] = softmax(local_income, self.softmax_temperature)
                 
            
        return reward
  
 
