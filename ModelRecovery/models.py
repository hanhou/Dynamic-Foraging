# =============================================================================
#  Models for fitting behavioral data
# =============================================================================
# == Simplified from foraging_testbed_models.py
#  
# = Forager types:
#   1. Special foragers
#       1). 'Random'
#       2). 'LossCounting': switch to another option when loss count exceeds a threshold drawn from Gaussian [from Shahidi 2019]
#           - 3.1: loss_count_threshold = inf --> Always One Side
#           - 3.2: loss_count_threshold = 1 --> win-stay-lose-switch
#           - 3.3: loss_count_threshold = 0 --> Always switch
#       3). 'IdealGreedy': knows p_reward and always chooses the largest one
#
#   2. NLP-like foragers
#       1). 'Sugrue2004':        income  ->   exp filter   ->  fractional                   -> epsilon-Poisson (epsilon = 0 in their paper; I found it essential)
#       2). 'Corrado2005':     income  ->  2-exp filter  ->  softmax ( = diff + sigmoid)  -> epsilon-Poisson (epsilon = 0 in their paper; has the same effect as tau_long??)
#       3). 'Iigaya2019':      income  ->  2-exp filter  ->  fractional                   -> epsilon-Poisson (epsilon = 0 in their paper; has the same effect as tau_long??)
#
#   3. RL-like foragers
#       1). 'SuttonBartoRLBook': return  ->   exp filter                                    -> epsilon-greedy  (epsilon > 0 is essential)
#       2). 'Bari2019':        return/income  ->   exp filter (both forgetting)   -> softmax     -> epsilon-Poisson (epsilon = 0 in their paper, no necessary)
#       3). 'Hattori2019':     return/income  ->   exp filter (choice-dependent forgetting, reward-dependent step_size)  -> softmax  -> epsilon-Poisson (epsilon = 0 in their paper; no necessary)
#
#   
#
# Feb 2020, Han Hou (houhan@gmail.com) @ Janelia 
# Svoboda & Li lab
# =============================================================================


import numpy as np
from scipy.stats import norm

LEFT = 0
RIGHT = 1

global_block_size_mean = 80
global_block_size_sd = 20

softmax = lambda x, softmax_temperature: np.exp(x/softmax_temperature)/np.sum(np.exp(x/softmax_temperature))  # Accept np.arrays


class BanditModels:
    
    # @K_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations
    
    def __init__(self, forager = 'SuttonBartoRLBook', K_arm = 2, n_trials = 1000, if_baited = True,  
                 epsilon = 0,          # Essential for 'SuttonBartoRLBook', 'Sugrue 2004', 'Iigaya2019'. See notes above.
                 softmax_temperature = np.nan,   # For 'Bari2019', 'Hattori2019','Corrado2005'
                 
                 taus = 20,        # For 'Sugrue2004' (only one tau is needed), 'Corrado2005', 'Iigaya2019'. Could be any number of taus, e.g., [2,15]
                 w_taus = 1,       # For 'Sugrue2004' (w_tau = 1), 'Corrado2005', 'Iigaya2019'.              Could be any number of taus, e.g., [0.3, 0.7]
                 
                 step_sizes = 0.1,      # For 'SuttonBartoRLBook'， 'Bari2019'， 'Hattori2019' (step_sizes = [unrewarded step_size, rewarded step_size]).
                 forget_rate = 0,      # For 'SuttonBartoRLBook' (= 0)， 'Bari2019' (= 1-Zeta)， 'Hattori2019' ( = unchosen_forget_rate).
                 
                 loss_count_threshold_mean = 3,   # For 'LossCounting' [from Shahidi 2019]
                 loss_count_threshold_std = 1,    # For 'LossCounting' [from Shahidi 2019]
                 
                 p_reward_seed_override = '',  # If true, use the same random seed for generating p_reward!!
                 p_reward_sum = 0.45,   # Gain of reward. Default = 0.45
                 p_reward_pairs = None,  # Full control of reward prob
                 
                 # Important for predictive fitting
                 # If not None, calculate predictive_choice_probs(t) based on fit_choice_history(0:t-1) and fit_reward_history(0:t-1) for negLL calculation.
                 fit_choice_history = None,
                 fit_reward_history = None,                 
                 
                 ):     

        self.forager = forager
        self.if_baited = if_baited
        self.epsilon = epsilon
        self.softmax_temperature = softmax_temperature
        self.loss_count_threshold_mean = loss_count_threshold_mean
        self.loss_count_threshold_std = loss_count_threshold_std
        self.p_reward_seed_override = p_reward_seed_override
        self.p_reward_sum = p_reward_sum
        self.p_reward_pairs = p_reward_pairs
        
        self.fit_choice_history = fit_choice_history
        self.fit_reward_history = fit_reward_history
        self.if_fit_mode = self.fit_choice_history is not None   # In some cases we just need fit_c to fit the model
        
        if self.if_fit_mode: 
            self.K, self.n_trials = np.shape(fit_reward_history)  # Use the targeted histories
        else:  # Backward compatibility
            self.K = K_arm
            self.n_trials = n_trials
        
        if forager == 'Sugrue2004':
            self.taus = [taus]
            self.w_taus = [w_taus]
        else:
            self.taus = taus
            self.w_taus = w_taus
        
        # Turn step_size and forget_rate into reward- and choice- dependent, respectively. (for 'Hattori2019')
        if forager == 'SuttonBartoRLBook':
            self.step_sizes = [step_sizes] * 2  # Replication
            self.forget_rates = [0, 0]   # Override
        elif forager == 'Bari2019':
            self.step_sizes = [step_sizes] * 2
            self.forget_rates = [forget_rate, forget_rate]
        elif forager == 'Hattori2019':
            self.step_sizes = step_sizes            # Should be [unrewarded step_size, rewarded_step_size] by itself
            self.forget_rates = [forget_rate, 0]   # Only unchosen target is forgetted
          
            
    def reset(self):
        
        #  print(self)
        
        # Initialization
        self.time = 0
        
        if self.if_fit_mode:  # Predictive mode
            self.predictive_choice_prob = np.zeros([self.K, self.n_trials])
            
        else:   # Generative mode
            self.choice_history = np.zeros([1,self.n_trials], dtype = int)  # Choice history
            self.reward_history = np.zeros([self.K, self.n_trials])    # Reward history, separated for each port (Corrado Newsome 2005)
        
            # Generate baiting prob in block structure
            self.generate_p_reward()
            
            # Prepare reward for the first trial
            # For example, [0,1] represents there is reward baited at the RIGHT but not LEFT port.
            self.reward_available = np.zeros([self.K, self.n_trials])    # Reward history, separated for each port (Corrado Newsome 2005)
            self.reward_available[:,0] = (np.random.uniform(0,1,self.K) < self.p_reward[:, self.time]).astype(int)
        
        # Forager-specific
        if self.forager in ['SuttonBartoRLBook', 'Bari2019', 'Hattori2019']:
            pass
        
        elif self.forager in ['Sugrue2004', 'Corrado2005', 'Iigaya2019']:
            # Compute the history filter. Compatible with any number of taus.
            reversed_t = np.flipud(np.arange(self.n_trials))  # Use the full length of the session just in case of an extremely large tau.
            self.history_filter = np.zeros_like(reversed_t).astype('float64')
            
            for tau, w_tau in zip(self.taus, self.w_taus):
                self.history_filter += w_tau * np.exp(-reversed_t / tau) / np.sum(np.exp(-reversed_t / tau))  # Note the normalization term (= tau when n -> inf.)
            
            self.q_estimation = np.zeros([self.K, self.n_trials]) # Estimation for each action (e.g., Q in Bari2019, L-stage scalar value in Corrado2005) 
            self.q_estimation[:] = 1/self.K   # To be strict
            
        elif self.forager in ['LossCounting']:
            # Initialize
            self.loss_count = np.zeros([1, self.n_trials]) 
            if not self.if_fit_mode:
                self.loss_threshold_this = np.random.normal(self.loss_count_threshold_mean, self.loss_count_threshold_std)

    def generate_p_reward(self, block_size_base = global_block_size_mean, 
                                block_size_sd = global_block_size_sd,
                                p_reward_pairs = [[.4,.05],[.3857,.0643],[.3375,.1125],[.225,.225]],  # (Bari-Cohen 2019)  
                         ):  
        
        # If para_optim, fix the random seed to ensure that p_reward schedule is fixed for all candidate parameters
        # However, we should make it random during a session (see the last line of this function)
        if self.p_reward_seed_override != '':
            np.random.seed(self.p_reward_seed_override)
            
        if self.p_reward_pairs == None:            
            p_reward_pairs = np.array(p_reward_pairs) / 0.45 * self.p_reward_sum
        else:  # Full override of p_reward
            p_reward_pairs = self.p_reward_pairs
                
        # Adapted from Marton's code
        n_trials_now = 0
        block_size = []  
        p_reward = np.zeros([2,self.n_trials])
        
        # Fill in trials until the required length
        while n_trials_now < self.n_trials:
        
            # Number of trials in each block (Gaussian distribution)
            # I treat p_reward[0,1] as the ENTIRE lists of reward probability. RIGHT = 0, LEFT = 1. HH
            n_trials_this_block = np.rint(np.random.normal(block_size_base, block_size_sd)).astype(int) 
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

        # We should make it random afterwards
        np.random.seed()

    def act(self):
        '''
        (Backward compatibility) If not in fitting mode, generate all choice on itself
        '''       
        if self.forager == 'Random': 
            choice = np.random.choice(self.K)
            
        elif self.forager == 'LossCounting':
            if self.time == 0:
                choice = np.random.choice(self.K)  # Random on the first trial
            else:
                # Retrieve the last choice
                last_choice = self.choice_history[0, self.time - 1]
                
                if self.loss_count[0, self.time] >= self.loss_threshold_this:
                    # Switch
                    choice = LEFT + RIGHT - last_choice
                    
                    # Reset loss counter threshold
                    self.loss_count[0, self.time] = - self.loss_count[0, self.time] # A flag of "switch happens here"
                    self.loss_threshold_this = np.random.normal(self.loss_count_threshold_mean, self.loss_count_threshold_std)
                else:
                    # Stay
                    choice = last_choice
            
        elif self.forager == 'IdealGreedy':
            choice = np.random.choice(np.where(self.p_reward[:,self.time] == self.p_reward[:,self.time].max())[0])
                
        else:
            if np.random.rand() < self.epsilon: 
                # Forced exploration with the prob. of epsilon (to avoid AlwaysLEFT/RIGHT in Sugrue2004...) or before some rewards are collected
                choice = np.random.choice(self.K)
                
            else:   # Forager-dependent
                if self.forager == 'SuttonBartoRLBook':   # Greedy
                    choice = np.random.choice(np.where(self.q_estimation[:, self.time] == self.q_estimation[:, self.time].max())[0])
                    
                elif self.forager in ['Sugrue2004', 'Corrado2005', 'Iigaya2019', 'Bari2019', 'Hattori2019' ]:   # Poisson
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
        
        self.time += 1   # Time ticks here !!!
        if self.time == self.n_trials: 
            return;   # Session terminates
        
        # For the next reward status, the "or" statement ensures the baiting property, gated by self.if_baited.
        self.reward_available[:, self.time] = np.logical_or(  reward_available_after_choice * self.if_baited,    
                                           np.random.uniform(0,1,self.K) < self.p_reward[:,self.time]).astype(int)  
            
                
        if self.forager in ['LossCounting']:
            if self.loss_count[0, self.time - 1] < 0:  # A switch just happened
                self.loss_count[0, self.time - 1] = - self.loss_count[0, self.time - 1]  # Back to normal (Note that this = 0 in Shahidi 2019)
                if reward:
                    self.loss_count[0, self.time] = 0
                else:
                    self.loss_count[0, self.time] = 1
            else:
                if reward:
                    self.loss_count[0, self.time] = self.loss_count[0, self.time - 1]
                else:
                    self.loss_count[0, self.time] = self.loss_count[0, self.time - 1] + 1
                
        elif self.forager in ['SuttonBartoRLBook', 'Bari2019', 'Hattori2019']:    
            
            # Reward-dependent step size ('Hattori2019')
            if reward:   
                step_size_this = self.step_sizes[1]
            else:
                step_size_this = self.step_sizes[0]
            
            # Choice-dependent forgetting rate ('Hattori2019')
            # Chosen:   Q(n+1) = (1- forget_rate_chosen) * Q(n) + step_size * (Reward - Q(n))
            self.q_estimation[choice, self.time] = (1 - self.forget_rates[1]) * self.q_estimation[choice, self.time - 1]  \
                                             + step_size_this * (reward - self.q_estimation[choice, self.time - 1])
                                                 
            # Unchosen: Q(n+1) = (1-forget_rate_unchosen) * Q(n)
            unchosen_idx = [cc for cc in range(self.K) if cc != choice]
            self.q_estimation[unchosen_idx, self.time] = (1 - self.forget_rates[0]) * self.q_estimation[unchosen_idx, self.time - 1] 
            
            # Softmax in 'Bari2019', 'Hattori2019'
            if self.forager in ['Bari2019', 'Hattori2019']:
                self.q_estimation[:, self.time] = softmax(self.q_estimation[:, self.time], self.softmax_temperature)
            
        elif self.forager in ['Sugrue2004', 'Iigaya2019']:
            # Fractional local income
            # Note: It's "income" because the following computations do not dependent on the current ~choice~.
            
            # 1. Local income = Reward history + exp filter in Sugrue or 2-exp filter in IIgaya
            valid_reward_history = self.reward_history[:, :self.time]   # History till now
            valid_filter = self.history_filter[-self.time:]    # Corresponding filter
            local_income = np.sum(valid_reward_history * valid_filter, axis = 1)
            
            # 2. Poisson choice probability = Fractional local income
            if np.sum(local_income) == 0:
                # 50%-to-50%
                self.q_estimation[:, self.time] = [1/self.K] * self.K
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
    
    def act_predict(self):
        '''
        In fitting mode, predict next choice probability using fit_choice_history and fit_reward_history            
        '''
        if self.time == 0:    # For all fittings, let the first trial random 
            self.predictive_choice_prob[:, self.time] = 1/self.K
            return  
        
        if self.forager == 'Random': 
            self.predictive_choice_prob[:, self.time] = 1/self.K
            pass
            
        elif self.forager == 'LossCounting':
            
            # Retrieve the last choice
            last_choice = self.fit_choice_history[0, self.time - 1]
            
            # Predict this choice prob
            # To be general, and ensure that alway switch when mean = 0, std = 0
            prob_switch = norm.cdf(self.loss_count[0, self.time], self.loss_count_threshold_mean - 1e-6, self.loss_count_threshold_std + 1e-16)  
            
            # Choice prob [choice] = 1-prob_switch, [others] = prob_switch /(K-1). Assuming randomly switch to other alternatives
            self.predictive_choice_prob[:, self.time] = prob_switch /(self.K - 1)
            self.predictive_choice_prob[last_choice, self.time] = 1 - prob_switch
            
            # Using fit_choice to mark an actual switch
            if last_choice != self.fit_choice_history[0, self.time]:
                self.loss_count[0, self.time] = - self.loss_count[0, self.time] # A flag of "switch happens here"
            
        else:   #!!!
            if np.random.rand() < self.epsilon: 
                # Forced exploration with the prob. of epsilon (to avoid AlwaysLEFT/RIGHT in Sugrue2004...) or before some rewards are collected
                choice = np.random.choice(self.K)
                
            else:   # Forager-dependent
                if self.forager == 'SuttonBartoRLBook':   # Greedy
                    choice = np.random.choice(np.where(self.q_estimation[:, self.time] == self.q_estimation[:, self.time].max())[0])
                    
                elif self.forager in ['Sugrue2004', 'Corrado2005', 'Iigaya2019', 'Bari2019', 'Hattori2019' ]:   # Poisson
                    if np.random.rand() < self.q_estimation[LEFT, self.time]:
                        choice = LEFT
                    else:
                        choice = RIGHT
                
        return   # Nothing to return
    
    def step_predict(self):
           
        # =============================================================================
        #  NO NEED TO: Generate reward or make the state transition (i.e., prepare reward for the next trial)
        #  Instead, we retrieve choice and reward from the targeted fit_c and fit_r 
        # =============================================================================
        choice = self.fit_choice_history[0, self.time]
        reward = self.fit_reward_history[choice, self.time]        
        
        self.time += 1   # Time ticks here !!!
        
        if self.time == self.n_trials: 
            return;   # Session terminates
                
        if self.forager in ['LossCounting']:
            if self.loss_count[0, self.time - 1] < 0:  # A switch just happened (in targeted fit_choice_history)
                self.loss_count[0, self.time - 1] = - self.loss_count[0, self.time - 1]  # Back to normal (Note that this = 0 in Shahidi 2019)
                if reward:
                    self.loss_count[0, self.time] = 0
                else:
                    self.loss_count[0, self.time] = 1
            else:
                if reward:
                    self.loss_count[0, self.time] = self.loss_count[0, self.time - 1]
                else:
                    self.loss_count[0, self.time] = self.loss_count[0, self.time - 1] + 1
                
        elif self.forager in ['SuttonBartoRLBook', 'Bari2019', 'Hattori2019']:      #!!!
            
            # Reward-dependent step size ('Hattori2019')
            if reward:   
                step_size_this = self.step_sizes[1]
            else:
                step_size_this = self.step_sizes[0]
            
            # Choice-dependent forgetting rate ('Hattori2019')
            # Chosen:   Q(n+1) = (1- forget_rate_chosen) * Q(n) + step_size * (Reward - Q(n))
            self.q_estimation[choice, self.time] = (1 - self.forget_rates[1]) * self.q_estimation[choice, self.time - 1]  \
                                             + step_size_this * (reward - self.q_estimation[choice, self.time - 1])
                                                 
            # Unchosen: Q(n+1) = (1-forget_rate_unchosen) * Q(n)
            unchosen_idx = [cc for cc in range(self.K) if cc != choice]
            self.q_estimation[unchosen_idx, self.time] = (1 - self.forget_rates[0]) * self.q_estimation[unchosen_idx, self.time - 1] 
            
            # Softmax in 'Bari2019', 'Hattori2019'
            if self.forager in ['Bari2019', 'Hattori2019']:
                self.q_estimation[:, self.time] = softmax(self.q_estimation[:, self.time], self.softmax_temperature)
            
        elif self.forager in ['Sugrue2004', 'Iigaya2019']:  #!!!
            # Fractional local income
            # Note: It's "income" because the following computations do not dependent on the current ~choice~.
            
            # 1. Local income = Reward history + exp filter in Sugrue or 2-exp filter in IIgaya
            valid_reward_history = self.reward_history[:, :self.time]   # History till now
            valid_filter = self.history_filter[-self.time:]    # Corresponding filter
            local_income = np.sum(valid_reward_history * valid_filter, axis = 1)
            
            # 2. Poisson choice probability = Fractional local income
            if np.sum(local_income) == 0:
                # 50%-to-50%
                self.q_estimation[:, self.time] = [1/self.K] * self.K
            else:
                # Local fractional income
                self.q_estimation[:, self.time] = local_income / np.sum(local_income)
                
        elif self.forager == 'Corrado2005':   #!!!
            # Softmaxed local income
            
            # 1. Local income = Reward history + hyperbolic (2-exps) filter
            valid_reward_history = self.reward_history[:, :self.time]   # History till now
            valid_filter = self.history_filter[-self.time:]    # Corresponding filter
            local_income = np.sum(valid_reward_history * valid_filter, axis = 1)
            
            # 2. Poisson choice probability = Softmaxed local income (Note: Equivalent to "difference + sigmoid" in [Corrado etal 2005], for 2lp case)
            self.q_estimation[:, self.time] = softmax(local_income, self.softmax_temperature)
                 
        return reward
    
    def simulate(self):
        
        # =============================================================================
        # Simulate one session
        # =============================================================================
        self.reset()
        
        for t in range(self.n_trials):        
            
            if self.if_fit_mode:  # If in fitting mode, predict next choice probability using fit_choice_history and fit_reward_history
                self.act_predict()
                self.step_predict()  # Note that here updating of Qs only depends on the targeted fit_c and fit_r, but NOT the model's choice
                
            else:    # (Backward compatibility) If not in fitting mode, generate all choice on itself
                # Loop: (Act --> Reward & New state)
                action = self.act()
                self.step(action)


