# -*- coding: utf-8 -*-
"""
Created on Fri May  8 22:29:03 2020

@author: Han
"""

import numpy as np
import matplotlib.pyplot as plt

from utils.helper_func import choose_ps, softmax

class FullStateQ():

    def __init__(self, K_arm = 2, first_choice = None,
                 max_run_length = 10, 
                 discount_rate = 0.99,
                 
                 learn_rate = 0.1,
                 softmax_temperature = 0.1, 
                 epsilon = 0.1,
                 ):
        
        self.learn_rate = learn_rate
        self.softmax_temperature = softmax_temperature
        self.discount_rate = discount_rate
        self.epsilon = epsilon
        
        self._init_states(max_run_length, K_arm)
        
        if first_choice is None: 
            first_choice = np.random.choice(K_arm)
        self.current_state = self.states[first_choice, 0]  # Randomly initialize the first choice
        self.backup_SA = [self.current_state, -1]  # First trial is a STAY at first_choice
        
    def _init_states(self, max_run_length, K_arm):
        
        # Generate a K_arm * max_run_length numpy array of states
        max_run_length = int(np.ceil(max_run_length))
        
        self.states = np.zeros([K_arm, max_run_length], dtype=object)
        for k in range(K_arm):
            for r in range(max_run_length):
                self.states[k,r] = State(k, r)
                
        # Define possible transitions
        for k in range(K_arm):
            for r in range(max_run_length):
                for kk in range(K_arm):
                    # Leave: to any other arms
                    if k != kk: self.states[k, r].add_next_states([self.states[kk, 0]])
                
                if r < max_run_length-1: self.states[k, r].add_next_states([self.states[k, r+1]])  # Stay is always the last index
                
    def act(self):   # State transition
        next_state_idx = self.current_state.act_softmax(self.softmax_temperature)  # Next state index!!
        # next_state_idx = self.current_state.act_epsilon(self.epsilon)  # Next state index!!
        
        self.backup_SA = [self.current_state, next_state_idx]     # For one-step backup in Q-learning
        self.current_state = self.current_state.next_states[next_state_idx]
        choice = self.current_state.which[0]  # Return absolute choice! (LEFT/RIGHT)
        return choice  
        
    def update_Q(self, reward):    # Q-learning (off-policy TD-0 bootstrap)
        max_next_SAvalue_for_backup_state = np.max(self.current_state.Q)  # This makes it off-policy
        last_state, last_choice = self.backup_SA
        last_state.Q[last_choice] += self.learn_rate * (reward + self.discount_rate * max_next_SAvalue_for_backup_state - last_state.Q[last_choice])  # Q-learning

        # print('Last: ', last_state.which, '(updated); This: ', self.current_state.which)
        # print('----------------------------------')
        # print('Left, leave: ', [s.Q[0] for s in self.states[0,:]])
        # print('Right,leave: ', [s.Q[0] for s in self.states[1,:]])
        # print('Left, stay : ', [s.Q[1] for s in self.states[0,:]])
        # print('Right,stay : ', [s.Q[1] for s in self.states[1,:]])
        
        
    def plot_V(self, ax, time, reward):  # Plot value functions (V(s) = max Q(s,a))
        direction = ['LEFT', 'RIGHT']    
        decision = ['Leave', 'Stay']
        for d in [0,1]:
            for c in [0,1]:
                p = [s.Q[c] for s in self.states[d,:]]
                ax[c, d].cla()
                ax[c, d].bar(np.arange(len(p)), p)
                ax[c, d].set_title(direction[d] + ', ' + decision[c])
                ax[c, d].axhline(y=0)
            
        plt.ylim([-1,1])
        plt.annotate( '%s --> %s\nt = %g, reward = %g' % (self.backup_SA[0].which, self.current_state.which, time, reward), xy=(0, 0.8), fontsize = 9)
        plt.gcf().canvas.draw()
        plt.waitforbuttonpress()
        # plt.pause(0.1)
        
class State():   
    
    '''
    Define states. 
    Technically, they're "agent states" in the agent's mind, 
    but in the baiting problem, they're actually also the CORRECT state representation of the environment
    '''    
    def __init__(self, _k, _run_length):
        # Which state is this?
        self.which = [_k, _run_length] # To be pricise: run_length - 1
        
        self.Q = np.array([0.0, 0.0])   # Action values for [Leave, Stay] of this state. Initialization value could be considered as priors?
        self.next_states = []  # All possible next states (other instances of class State)
    
    def add_next_states(self, next_states):
        self.next_states.extend(next_states)
        
    def act_softmax(self, softmax_temp):  # Generate the next action using softmax(Q) policy
        next_state_idx = choose_ps(softmax(self.Q[:len(self.next_states)], softmax_temp))
        return next_state_idx  # Return the index of the next state

    def act_epsilon(self, epsilon):  # Generate the next action using epsilon-greedy (More exploratory)
        if np.random.rand() < epsilon:
            next_state_idx = np.random.choice(len(self.next_states))
        else:   # Greedy
            Q_available = self.Q[:len(self.next_states)]
            next_state_idx = np.random.choice(np.where(Q_available == Q_available.max())[0])
        return next_state_idx  # Return the index of the next state
