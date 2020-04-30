# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 21:33:33 2020

@author: Han
# """

import numpy as np
import multiprocessing as mp
import time
import sys, os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


from models.bandit_model_comparison import BanditModelComparison
from utils.plot_mice import plot_each_mice, analyze_runlength_Lau2005, plot_runlength_Lau2005, plot_example_sessions

def fit_each_mice(data, if_session_wise = False, if_verbose = True, file_name = '', pool = '', models = None, use_trials = None):
    choice = data.f.choice
    reward = data.f.reward
    p1 = data.f.p1
    p2 = data.f.p2
    session_num = data.f.session
    
    # -- Formating --
    # Remove ignores
    valid_trials = choice != 0
    
    choice_history = choice[valid_trials] - 1  # 1: LEFT, 2: RIGHT --> 0: LEFT, 1: RIGHT
    reward = reward[valid_trials]
    p_reward = np.vstack((p1[valid_trials],p2[valid_trials]))
    session_num = session_num[valid_trials]
    
    n_trials = len(choice_history)
    print('Total valid trials = %g' % n_trials)
    sys.stdout.flush()
    
    reward_history = np.zeros([2,n_trials])
    for c in (0,1):  
        reward_history[c, choice_history == c] = (reward[choice_history == c] > 0).astype(int)
    
    choice_history = np.array([choice_history])
    
    results_each_mice = {}
    
    # -- Model comparison for each session --
    if if_session_wise:
        
        model_comparison_session_wise = []
        
        unique_session = np.unique(session_num)
        
        for ss in tqdm(unique_session, desc = 'Session-wise', total = len(unique_session)):
            choice_history_this = choice_history[:, session_num == ss]
            reward_history_this = reward_history[:, session_num == ss]

            if use_trials is not None:
                choice_history_this = choice_history_this[:, use_trials]
                reward_history_this = reward_history_this[:, use_trials]
                
            model_comparison_this = BanditModelComparison(choice_history_this, reward_history_this, models = models)
            model_comparison_this.fit(pool = pool, plot_predictive = None, if_verbose = False) # Plot predictive traces for the 1st, 2nd, and 3rd models
            model_comparison_session_wise.append(model_comparison_this)
                
        results_each_mice['model_comparison_session_wise'] = model_comparison_session_wise
    
    # -- Model comparison for all trials --
    # For debugging    
    if use_trials is not None:
        choice_history = choice_history[:, use_trials]
        reward_history = reward_history[:, use_trials]
        p_reward = p_reward[:,use_trials]
        session_num = session_num[use_trials]
    
    print('Pooling all sessions: ', end='')
    start = time.time()
    model_comparison_grand = BanditModelComparison(choice_history, reward_history, p_reward = p_reward, session_num = session_num, models = models)
    model_comparison_grand.fit(pool = pool, plot_predictive = None if if_session_wise else [1,2,3], if_verbose = if_verbose) # Plot predictive traces for the 1st, 2nd, and 3rd models
    print(' Done in %g secs' % (time.time() - start))
    
    if if_verbose:
        model_comparison_grand.show()
        model_comparison_grand.plot()
    
    results_each_mice['model_comparison_grand'] = model_comparison_grand    
    
    return results_each_mice

def fit_all_mice(path, save_prefix = 'model_comparison', pool = '', models = None):
    # -- Find all files --
    start_all = time.time()
    for r, _, f in os.walk(path):
        for file in f:
            data = np.load(os.path.join(r, file))
            print('=== Mice %s ===' % file)
            start = time.time()
            
            # Do it
            try:
                results_each_mice = fit_each_mice(data, file_name = file, pool = pool, models = models, if_session_wise = True, if_verbose = False)
                np.savez_compressed( path + save_prefix + '_%s' % file, results_each_mice = results_each_mice)
                print('Mice %s done in %g mins!\n' % (file, (time.time() - start)/60))
            except:
                print('SOMETHING WENT WRONG!!')
                
    print('\n ALL FINISHED IN %g hrs!' % ((time.time() - start_all)/3600) )

def combine_each_model_comparison(objectA, objectB):
    '''
    Combine two BanditModelComparison objects

    '''
    
    # Confirm they were from the same dataset
    assert(np.all(objectA.fit_choice_history == objectB.fit_choice_history))
    assert(np.all(objectA.fit_reward_history == objectB.fit_reward_history))

    # -- Add info in objectB into object A --
    objectA.models.extend(objectB.models) 
    objectA.results_raw.extend(objectB.results_raw) 
    
    new_pd = pd.concat([objectA.results, objectB.results], axis = 0)

    # -- Update table --
    delta_AIC = new_pd.AIC - np.min(new_pd.AIC) 
    delta_BIC = new_pd.BIC - np.min(new_pd.BIC)

    # Relative likelihood = Bayes factor = p_model/p_best = exp( - delta_AIC / 2)
    new_pd['relative_likelihood_AIC'] = np.exp( - delta_AIC / 2)
    new_pd['relative_likelihood_BIC'] = np.exp( - delta_BIC / 2)

    # Model weight = Relative likelihood / sum(Relative likelihood)
    new_pd['model_weight_AIC'] = new_pd['relative_likelihood_AIC'] / np.sum(new_pd['relative_likelihood_AIC'])
    new_pd['model_weight_BIC'] = new_pd['relative_likelihood_BIC'] / np.sum(new_pd['relative_likelihood_BIC'])
    
    # log_10 (Bayes factor) = log_10 (exp( - delta_AIC / 2)) = (-delta_AIC / 2) / log(10)
    new_pd['log10_BF_AIC'] = - delta_AIC/2 / np.log(10) # Calculate log10(Bayes factor) (relative likelihood)
    new_pd['log10_BF_BIC'] = - delta_BIC/2 / np.log(10) # Calculate log10(Bayes factor) (relative likelihood)
    
    new_pd['best_model_AIC'] = (new_pd.AIC == np.min(new_pd.AIC)).astype(int)
    new_pd['best_model_BIC'] = (new_pd.BIC == np.min(new_pd.BIC)).astype(int)
    
    new_pd.index = range(1,1+len(new_pd))
    
    # Update notations
    para_notation_with_best_fit = []
    for i, row in new_pd.iterrows():
        para_notation_with_best_fit.append('('+str(i)+') '+row.para_notation + '\n' + str(np.round(row.para_fitted,2)))

    new_pd['para_notation_with_best_fit'] = para_notation_with_best_fit

    objectA.results = new_pd
    objectA.results_sort = new_pd.sort_values(by='AIC')

    return objectA

def combine_group_results(raw_path = "..\\export\\", result_path = "..\\results\\model_comparison\\",
                          combine_prefix = ['model_comparison_', 'model_comparison_no_bias_'], save_prefix = 'model_comparison_15_'):
    '''
    Combine TWO runs of model comparison
    '''
    for r, _, f in os.walk(raw_path):
        for file in f:

            data_A = np.load(result_path + combine_prefix[0] + file, allow_pickle=True)
            data_A = data_A.f.results_each_mice.item()
            
            data_B = np.load(result_path + combine_prefix[1] + file, allow_pickle=True)
            data_B = data_B.f.results_each_mice.item()
            
            new_grand_mc = combine_each_model_comparison(data_A['model_comparison_grand'], data_B['model_comparison_grand'])
            
            new_session_wise_mc = []
            for AA, BB in zip(data_A['model_comparison_session_wise'], data_B['model_comparison_session_wise']):
                new_session_wise_mc.append(combine_each_model_comparison(AA, BB))
                
            # -- Save data --
            results_each_mice = {'model_comparison_grand': new_grand_mc, 'model_comparison_session_wise': new_session_wise_mc}
            np.savez_compressed( result_path + save_prefix + file, results_each_mice = results_each_mice)
            print('%s + %s: Combined!' %(combine_prefix[0] + file, combine_prefix[1] + file))
            
def get_p_hat_greedy(p_reward):
    p_max = np.max(p_reward, axis=0)
    p_min = np.min(p_reward, axis=0)
    
    # p_min > 0
    m_star = np.floor(np.log(1-p_max[p_min > 0])/np.log(1-p_min[p_min > 0]))
    p_star = p_max[p_min > 0] + (1-(1-p_min[p_min > 0])**(m_star + 1)-p_max[p_min > 0]**2)/(m_star+1)
    
    # p_min == 0
    p_star = np.hstack([p_star, p_max[p_min == 0]])
    p_star_aver = np.mean(p_star)
    
    return p_star_aver

def process_each_mice(data, file, if_plot_each_mice):

    # === 1.1 Overall result ===
    grand_result = data['model_comparison_grand'].results
    overall_best = np.where(grand_result['best_model_AIC'])[0][0] + 1

    # === Session-wise ===
    sessionwise_result =  data['model_comparison_session_wise']
    n_session = len(sessionwise_result)
    n_models = len(grand_result)
    
    # --- Reorganize data ---
    group_result = {'n_trials': np.zeros((1,n_session))}
    group_result['session_number'] = np.unique(data['model_comparison_grand'].session_num)
    group_result['session_best'] = np.zeros(n_session).astype(int)
    group_result['prediction_accuracy_NONCV'] = np.zeros(n_session)
    group_result['prediction_accuracy_Sugrue_NONCV'] = np.zeros(n_session)
    group_result['foraging_efficiency'] = np.zeros(n_session)
    group_result['raw_data'] = data
    group_result['file'] = file
    
    group_result['xlabel'] = []
    for ss,this_mc in enumerate(sessionwise_result):
        group_result['n_trials'][0,ss] = this_mc.n_trials
        group_result['xlabel'].append(str(group_result['session_number'][ss]) + '\n('+ str(this_mc.n_trials) +')')
        
        # Prediction accuracy of the best model (NOT cross-validated!!)
        group_result['session_best'][ss] = (np.where(this_mc.results['best_model_AIC'])[0][0] + 1)
        this_predictive_choice_prob =  this_mc.results_raw[group_result['session_best'][ss] - 1].predictive_choice_prob
        this_predictive_choice = np.argmax(this_predictive_choice_prob, axis = 0)
        group_result['prediction_accuracy_NONCV'][ss] = np.sum(this_predictive_choice == this_mc.fit_choice_history) / this_mc.n_trials

        # Prediction accuracy of Sugrue (Marton's question)
        this_predictive_choice_prob =  this_mc.results_raw[3].predictive_choice_prob
        this_predictive_choice = np.argmax(this_predictive_choice_prob, axis = 0)
        group_result['prediction_accuracy_Sugrue_NONCV'][ss] = np.sum(this_predictive_choice == this_mc.fit_choice_history) / this_mc.n_trials

        # Calculate foraging efficiency
        p_reward_this = data['model_comparison_grand'].p_reward[:, data['model_comparison_grand'].session_num == group_result['session_number'][ss]]
        reward_rate_p_hat_greedy = get_p_hat_greedy(p_reward_this)
        group_result['foraging_efficiency'][ss] = (np.sum(this_mc.fit_reward_history)/group_result['n_trials'][0,ss]) / reward_rate_p_hat_greedy

    # Grand stuffs
    this_predictive_choice_prob_grand =  data['model_comparison_grand'].results_raw[overall_best - 1].predictive_choice_prob
    this_predictive_choice_grand = np.argmax(this_predictive_choice_prob_grand, axis = 0)
    group_result['prediction_accuracy_NONCV_grand'] = np.sum(this_predictive_choice_grand == data['model_comparison_grand'].fit_choice_history) / data['model_comparison_grand'].n_trials 
    
    p_reward_grand = data['model_comparison_grand'].p_reward
    reward_rate_p_hat_greedy = get_p_hat_greedy(p_reward_grand)
    group_result['foraging_efficiency_grand'] = (np.sum(data['model_comparison_grand'].fit_reward_history)/data['model_comparison_grand'].n_trials) /\
                                                reward_rate_p_hat_greedy


    # Iteratively copy data
    things_of_interest = ['model_weight_AIC','AIC', 'LPT_AIC']
    for pp in things_of_interest:
        group_result[pp] = np.zeros((n_models, n_session))
        for ss,this_mc in enumerate(sessionwise_result):
            group_result[pp][:, ss] = this_mc.results[pp]
            
    standardRL_idx = 12  # RW1972-softmax-noBias (0 of Fig.1I of Hattori 2019)    
    group_result['delta_AIC'] = group_result['AIC'] - group_result['AIC'][standardRL_idx - 1]
    group_result['delta_AIC_grand'] = grand_result['AIC'] -  grand_result['AIC'][standardRL_idx]
    group_result['LPT_AIC_grand']  = grand_result['LPT_AIC']
    group_result['fitted_paras_grand'] = grand_result['para_fitted'].iloc[overall_best-1]
    
    # -- Fitting results --
    data['model_comparison_grand'].plot_predictive = [1,2,3]
            
    # -- 2.1 deltaAIC relative to RW1972-softmax-noBias (Fig.1I of Hattori 2019) --        
    plot_models = np.array([13, 6, 15, 8])
    group_result['delta_AIC'] = group_result['delta_AIC'][plot_models - 1,:]
    group_result['delta_AIC_para_notation'] = grand_result['para_notation'].iloc[plot_models-1]
    group_result['delta_AIC_grand'] = group_result['delta_AIC_grand'][plot_models - 1]
    
    # -- 3. Fitted paras of the overall best model --
    fitted_para_names = np.array(grand_result['para_notation'].iloc[overall_best-1].split(','))
    group_result['fitted_para_names'] = fitted_para_names
    group_result['fitted_paras'] = np.zeros((len(fitted_para_names), n_session))
    for ss,this_mc in enumerate(sessionwise_result):
        group_result['fitted_paras'][:,ss] = this_mc.results['para_fitted'].iloc[overall_best-1]
        
    # -- 4. Prediction accuracy of the bias term only (Hattori Fig.1J) --
    # When softmax only have a bias term, choice_right_prob = 1/(1+exp(delta/sigma)), choice_left_prob = 1/(1+exp(-delta/sigma))
    # Therefore, the prediction accuracy = proportion of left (right) choices, if delta > 0 (delta <= 0)
    group_result['prediction_accuracy_bias_only'] = np.zeros(n_session)
    for ss,this_mc in enumerate(sessionwise_result):
        bias_this = group_result['fitted_paras'][fitted_para_names == ' $b_L$', ss]
        which_largest = int(bias_this <= 0) # If bias_this < 0, bias predicts all rightward choices
        group_result['prediction_accuracy_bias_only'][ss] = np.sum(this_mc.fit_choice_history == which_largest) / this_mc.n_trials
        
    bias_grand = group_result['fitted_paras_grand'][fitted_para_names == ' $b_L$']
    which_largest_grand = int(bias_grand <= 0)
    group_result['prediction_accuracy_bias_only_grand'] = np.sum(data['model_comparison_grand'].fit_choice_history == which_largest_grand) /\
                                                          data['model_comparison_grand'].n_trials 

    if if_plot_each_mice:
        plot_each_mice(group_result)
    
    return group_result


def process_all_mice(result_path = "..\\results\\model_comparison\\", combine_prefix = 'model_comparison_15_', mice_select = '', 
                  group_results_name_to_save = 'temp.npz', if_plot_each_mice = True):
    
    #%%
    listOfFiles = os.listdir(result_path)
    
    results_all_mice = pd.DataFrame()
    n_mice = 0
    
    for file in listOfFiles:
        
        # -- Screening of file name --
        if mice_select != '':
            skip = True
            for ss in mice_select:
                if ss in file: 
                    skip = False
                    break
            if skip: continue  # Pass this file
        
        if combine_prefix not in file: continue # Pass this file
        
        # print(file)
        n_mice += 1
        data = np.load(result_path + file, allow_pickle=True)
        data = data.f.results_each_mice.item()
        
        group_result_this = process_each_mice(data, file, if_plot_each_mice = if_plot_each_mice)
        df_this = pd.DataFrame({'mice': file.replace(combine_prefix,'').replace('.npz',''),
                                'session_idx': np.arange(len(group_result_this['session_number'])) + 1,
                                'session_number': group_result_this['session_number'],
                                 })
        df_this['prediction_accuracy_NONCV'] = group_result_this['prediction_accuracy_NONCV']
        df_this['prediction_accuracy_bias_only'] = group_result_this['prediction_accuracy_bias_only']
        
        df_this['prediction_accuracy_Sugrue_NONCV'] = group_result_this['prediction_accuracy_Sugrue_NONCV']
        
        df_this['foraging_efficiency'] = group_result_this['foraging_efficiency']
        df_this['n_trials'] = group_result_this['n_trials'][0]
        df_this = pd.concat([df_this, pd.DataFrame(group_result_this['fitted_paras'].T, columns = group_result_this['fitted_para_names'])], axis = 1)
        df_this = pd.concat([df_this, pd.DataFrame(group_result_this['delta_AIC'].T, columns = group_result_this['delta_AIC_para_notation'])], axis = 1)
        
        # Turn mine definition of bias (outside softmax) into Hattori's definition (inside softmax) ??? 
        # But I think they actually used my definition? Otherwise their biases are huge (beta * beta_deltaQ is much larger than beta)...
        # So I don't want to do any modifications
        # df_this[' $b_L$'] = df_this[' $b_L$'] * df_this[' $\sigma$']
        
        # Save dataframe of this mice into a HUGE dataframe
        results_all_mice = results_all_mice.append(df_this)
        
    # Add some more stuffs for convenience
    group_results = {'results_all_mice': results_all_mice}     
    group_results['delta_AIC_para_notation'] = group_result_this['delta_AIC_para_notation']
    group_results['fitted_para_names'] = group_result_this['fitted_para_names']
    group_results['n_mice'] = n_mice 
    
    # Save group results to file
    np.savez_compressed(result_path + group_results_name_to_save, group_results = group_results)
    
    # results_all_mice.to_pickle(result_path + 'results_all_mice.pkl')
    print('Group results saved: %s!' %(result_path + group_results_name_to_save))
        

def analyze_runlength(result_path = "..\\results\\model_comparison\\", combine_prefix = 'model_comparison_15_', 
                          group_results_name = 'group_results.npz', mice_of_interest = ['FOR05', 'FOR06'], 
                          efficiency_partitions = [30, 30],  block_partitions = [70, 70], if_first_plot = True):
    sns.set()
    
    # Load dataframe
    data = np.load(result_path + group_results_name, allow_pickle=True)
    group_results = data.f.group_results.item()
    results_all_mice = group_results['results_all_mice'] 
    
    for mouse in mice_of_interest:
        
        # Load raw data
        data_raw = np.load(result_path + combine_prefix + mouse + '.npz', allow_pickle=True)
        data_raw = data_raw.f.results_each_mice.item()
        
        df_this = results_all_mice[results_all_mice.mice == mouse].copy()
        df_this[['foraging_efficiency', 'prediction_accuracy_NONCV', 'prediction_accuracy_bias_only', 'prediction_accuracy_Sugrue_NONCV']] *= 100

        efficiency_thres = np.percentile(df_this.foraging_efficiency, [100-efficiency_partitions[0], efficiency_partitions[1]])

        #%% Plot foraging histogram 
        if if_first_plot: 
            
            
            x = df_this.prediction_accuracy_NONCV
            y = df_this.foraging_efficiency
            (r, p) = pearsonr(x, y)
  
            g = sns.jointplot(x="prediction_accuracy_NONCV", y="foraging_efficiency", data = df_this.sort_values(by = 'session_number'), 
                              kind="reg", color="b", marginal_kws = {'bins':20,'color':'k'}, joint_kws = {'marker':'', 'color':'k', 
                                                                                                          'label':'r$^2$ = %.3f, p = %.3f'%(r**2,p)})
            
            palette = sns.color_palette("RdYlGn", len(df_this))
            
            g.plot_joint(plt.scatter, color = palette, sizes = df_this.n_trials**2 / 3000, alpha = 0.7)
            plt.legend()
            ax = plt.gca()
            ax.axvline(50, c='k', ls='--')
            ax.axhline(100, c='k', ls='--')
            
            ax.axhline(efficiency_thres[1], c='r', ls='-.', lw = 2)        
            ax.axhline(efficiency_thres[0], c='g', ls='-.', lw = 2)
            plt.gcf().text(0.01, 0.95, mouse)

        #%% Get grand runlength (Lau)
        good_session_idxs = df_this[df_this.foraging_efficiency >= efficiency_thres[0]].session_idx
        bad_session_idxs = df_this[df_this.foraging_efficiency <= efficiency_thres[1]].session_idx
        
        grand_session_idxs = [good_session_idxs, bad_session_idxs]
        grand_session_idxs_markers = [mouse + ' best %g%% sessions (n = %g)' % (efficiency_partitions[0], len(good_session_idxs)), 
                                      mouse + ' worst %g%% sessions (n = %g)' % (efficiency_partitions[1], len(bad_session_idxs))]
        
        for this_session_idxs, this_marker in zip(grand_session_idxs, grand_session_idxs_markers):
            
            df_run_length_Lau_all = [pd.DataFrame(), pd.DataFrame()] # First and last trials in each block
            
            for this_idx in this_session_idxs:
                #%%
                this_class = data_raw['model_comparison_session_wise'][this_idx - 1]
                this_session_num = df_this[df_this.session_idx == this_idx].session_number.values
                
                p_reward = data_raw['model_comparison_grand'].p_reward[:, data_raw['model_comparison_grand'].session_num == this_session_num]
                
                # Runlength analysis
                this_df_run_length_Lau = analyze_runlength_Lau2005(this_class.fit_choice_history, p_reward, block_partitions = block_partitions)
                
                for i in [0,1]:
                    df_run_length_Lau_all[i] = df_run_length_Lau_all[i].append(this_df_run_length_Lau[i])
                
            fig = plot_runlength_Lau2005(df_run_length_Lau_all, block_partitions)
            fig.text(0.1, 0.92, this_marker + ', mean foraging eff. = %g%%, %g blocks' %\
                           (np.mean(df_this.foraging_efficiency[this_session_idxs - 1]), len(df_run_length_Lau_all[0])), fontsize = 15)
            plt.show()
            
def analyze_runlength_of_models(block_partitions = [50,50]):   # Runlength analyses for Hattori and Ideal-optimal etc. as a comparison

    from utils.run_model_recovery import generate_fake_data
    from utils.plot_fitting import plot_session_lightweight
    
    # Best Hattori 2019 model (under the default schedules of Bari 2019)
    choice_history, reward_history, p_reward = generate_fake_data('Hattori2019', ['learn_rate_rew','learn_rate_unrew', 'forget_rate','softmax_temperature'], 
                                                      [0.23392543, 0.318161268, 0.00343416, 0.22028081], n_trials = 10000)
    
    foraging_efficiency = np.sum(reward_history) / np.shape(reward_history)[1] / get_p_hat_greedy(p_reward) * 100
    plot_session_lightweight([choice_history, reward_history, p_reward], smooth_factor = 1)
    plt.gca().set_title('Best Hattori, foraging eff. = %g%%'%foraging_efficiency)
    plt.gca().set_xlim([0,200])
    
    run_length_Lau = analyze_runlength_Lau2005(choice_history, p_reward, block_partitions = block_partitions)
    plot_runlength_Lau2005(run_length_Lau, block_partitions)
    plt.gcf().text(0.02,0.92,'Best Hattori, foraging eff. = %g%%'%foraging_efficiency)
    
    # Best Hattori 2019 model (under the default schedules of Bari 2019)
    choice_history, reward_history, p_reward = generate_fake_data('IdealpHatGreedy', [],[], n_trials = 10000)
    
    foraging_efficiency = np.sum(reward_history) / np.shape(reward_history)[1] / get_p_hat_greedy(p_reward) * 100
    plot_session_lightweight([choice_history, reward_history, p_reward], smooth_factor = 1)
    plt.gca().set_title('Ideal-$\\hat{p}$-greedy, foraging eff. = %g%%'%foraging_efficiency)
    plt.gca().set_xlim([0,200])
    
    run_length_Lau = analyze_runlength_Lau2005(choice_history, p_reward, block_partitions = block_partitions)
    plot_runlength_Lau2005(run_length_Lau, block_partitions)
    
    plot_runlength_Lau2005(run_length_Lau, block_partitions)
    plt.gcf().text(0.02,0.92,'Ideal-$\\hat{p}$-greedy, foraging eff. = %g%%'%foraging_efficiency)
  
                    
#%%        
if __name__ == '__main__':
    
    n_worker = 8
    # pool = mp.Pool(processes = n_worker)
    
    # ---
    # data = np.load("..\\export\\FOR01.npz")
    # model_comparison = fit_each_mice(data, pool = pool, models = [1,9], use_trials = np.r_[0:500])
    
    # --- Fit all mice, session-wise and pooling
    # fit_all_mice(path = '..\\export\\', save_prefix = 'model_comparison_no_bias' , models = [1,9,10,11,12,13,14,15], pool = pool)
    
    # --- Combine different runs ---
    # combine_group_results()
    
    # --- Plot all results ---
    # process_all_mice(result_path = "..\\results\\model_comparison\\", combine_prefix = 'model_comparison_15_', group_results_name_to_save = 'temp.npz', if_plot_each_mice = False)
    # plot_group_results(group_results_name = 'temp.npz', average_session_number_range = [0,20])
    
    # --- Example sessions ---
    # plot_example_sessions(group_results_name = 'temp.npz', session_of_interest = [['FOR05', 33]])
    
    analyze_runlength(mice_of_interest = ['FOR05', 'FOR06'], efficiency_partitions = [20, 20], block_partitions = [30, 30])
    # analyze_runlength(efficiency_partitions = [20, 20], block_partitions = [50, 50], if_first_plot = False)
    
    # analyze_runlength_of_models()
    
    # pool.close()   # Just a good practice
    # pool.join()
