# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:03:28 2020

@author: Han
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statannot import add_stat_annotation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from utils.plot_fitting import set_label

plt.rcParams.update({'figure.max_open_warning': 0})


def plot_each_mice(group_result):
    
    # Unpack data
    file = group_result['file']
    data = group_result['raw_data']

    grand_result = data['model_comparison_grand'].results
    sessionwise_result =  data['model_comparison_session_wise']
    overall_best = np.where(grand_result['best_model_AIC'])[0][0] + 1
    session_best_matched = group_result['session_best'] == overall_best
    model_notations = grand_result['para_notation_with_best_fit']
    
    fitted_para_names = np.array(grand_result['para_notation'].iloc[overall_best-1].split(','))

    #%%
    # plt.close('all')
    sns.set()

    plt.rcParams.update({'font.size': 8, 'figure.dpi': 150})

    fig = plt.figure(figsize=(15, 9), dpi = 150)
    gs = GridSpec(1, 20, wspace = 0.1, bottom = 0.15, top = 0.9, left = 0.15, right = 0.95)
    fig.text(0.01,0.95,'%s' % (file), fontsize = 20)
    
    # === 1.1 Overall result ===
    ax = fig.add_subplot(gs[0, 0:1])
    sns.heatmap(grand_result[['model_weight_AIC']], annot = True, fmt=".2f", square = False, cbar = False, cbar_ax = [0,1])
    
    patch = Rectangle((0, np.where(grand_result['best_model_AIC'])[0]),1,1, color = 'dodgerblue', linewidth = 4, fill= False)
    ax.add_artist(patch)
    
    set_label(ax, 1,0, model_notations)
    ax.set_xticklabels(['Overall\n(' + str(data['model_comparison_grand'].n_trials) + ')'])

    # --- 1.2 Session-wise model weight ---
    ax = fig.add_subplot(gs[0, 1: round(20)])
    sns.heatmap(group_result['model_weight_AIC'], annot = True, fmt=".2f", square = False, cbar = False, cbar_ax = [0,1])
    ax.set_yticklabels('')
    ax.set_xticklabels(group_result['xlabel'])

    for ss,this_mc in enumerate(sessionwise_result):
        patch = Rectangle((ss, group_result['session_best'][ss] - 1),1,1, color = 'dodgerblue', linewidth = 4, fill= False)
        ax.add_artist(patch)
        
    # -- Fitting results --
    data['model_comparison_grand'].plot_predictive_choice()
    
    # -- 2.1 deltaAIC relative to RW1972-softmax-noBias (Fig.1I of Hattori 2019) --        

    plot_colors = ['orange', 'b', 'g', 'r'] # Orange, blue, green, red
    marker_sizes = (group_result["n_trials"]/100 * 2)**2
    
    fig = plt.figure(figsize=(15, 5), dpi = 150)
    gs = GridSpec(1, 20, wspace = 0.2, bottom = 0.15, top = 0.9, left = 0.07, right = 0.95)
    
    ax = fig.add_subplot(gs[0, 0:9])    
    plt.axhline(0, c = 'k', ls = '--')
    for mm, cc in enumerate(plot_colors):
        
        # plt.plot(group_result['session_number'].T, group_result['delta_AIC'].T)
        x = group_result['session_number']
        y = group_result['delta_AIC'][mm,:]
        ax.plot(x, y, color = cc, label = group_result['delta_AIC_para_notation'].iloc[mm], linewidth = 0.7)
        
        ax.scatter(x[session_best_matched], y[session_best_matched], color = cc, s = marker_sizes, alpha = 0.9)
        ax.scatter(x[np.logical_not(session_best_matched)], y[np.logical_not(session_best_matched)], facecolors='none', edgecolors = cc, s = marker_sizes, alpha = 0.7)
        
    ax.text(min(plt.xlim()),10,'(12) RW1972-noBias')
    ax.set_xlabel('Session number')
    ax.set_ylabel('$\\Delta$ AIC')
    ax.legend()    
    
    # -- 2.2 Likelihood per trial and prediction accuracy (TBD) --
    ax_grand = fig.add_subplot(gs[0, 10])    
    plt.axhline(0.5, c = 'k', ls = '--')
    
    # > Overall
    # # LPT_AIC
    # sns.pointplot(data = group_result['LPT_AIC'][overall_best-1,:], ax = ax_grand, color = 'b', ci = 68)
    # ax_grand.plot(-1, group_result['LPT_AIC_grand'][overall_best-1], marker = 's', markersize = 13, color = 'b')
    
    # Prediction accuracy NONCV
    sns.pointplot(data = group_result['prediction_accuracy_NONCV'], ax = ax_grand, color = 'black', ci = 68)
    ax_grand.plot(-1, group_result['prediction_accuracy_NONCV_grand'], marker = 's', markersize = 13, color = 'black')
    
    # Prediction accuracy bias only
    sns.pointplot(data = group_result['prediction_accuracy_bias_only'], ax = ax_grand, color = 'r', ci = 68)
    ax_grand.plot(-1, group_result['prediction_accuracy_bias_only_grand'], marker = 's', markersize = 13, color = 'r')

    # > Session-wise
    ax = fig.add_subplot(gs[0, 11:20], sharey = ax_grand)    
    plt.axhline(0.5, c = 'k', ls = '--')
    
    # # LPT_AIC
    # x = group_result['session_number']
    # y = group_result['LPT_AIC'][overall_best-1,:]
    # plt.plot(x, y, 'b',label = 'likelihood per trial (AIC)', linewidth = 0.7)
    # plt.scatter(x[session_best_matched], y[session_best_matched], color = 'b', s = marker_sizes, alpha = 0.9)
    # plt.scatter(x[np.logical_not(session_best_matched)], y[np.logical_not(session_best_matched)], 
    #             facecolors='none', edgecolors = 'b', s = marker_sizes, alpha = 0.7)

    # Prediction accuracy NONCV
    y = group_result['prediction_accuracy_NONCV']
    plt.plot(x, y, 'k',label = 'prediction accuracy of the best model', linewidth = 0.7)
    plt.scatter(x[session_best_matched], y[session_best_matched], color = 'k', s = marker_sizes, alpha = 0.9, label = 'session = overall best')
    plt.scatter(x[np.logical_not(session_best_matched)], y[np.logical_not(session_best_matched)], 
                facecolors='none', edgecolors = 'k', s = marker_sizes, alpha = 0.7, label = 'session $\\neq$ overall best')
    
    # Prediction accuracy bias only
    y = group_result['prediction_accuracy_bias_only']
    plt.plot(x, y, 'r', ls = '--', label = 'prediction accuracy of bias only', linewidth = 0.7)
    plt.scatter(x[session_best_matched], y[session_best_matched], color = 'r', s = marker_sizes, alpha = 0.9)
    plt.scatter(x[np.logical_not(session_best_matched)], y[np.logical_not(session_best_matched)], 
                facecolors='none', edgecolors = 'r', s = marker_sizes, alpha = 0.7)

    ax_grand.set_xticks([-1,0])
    ax_grand.set_xlim([-1.5,.5])
    ax_grand.set_xticklabels(['Overall','mean'], rotation = 45)
    # ax_grand.set_ylabel('Likelihood per trial (AIC)')
    plt.setp(ax.get_yticklabels(), visible=False)
    
    plt.title('Overall best: %s {%s}' % (grand_result['model'].iloc[overall_best-1], grand_result['para_notation'].iloc[overall_best-1]))
    plt.xlabel('Session number')
    plt.legend()    

    # -- 3 Fitted paras of the overall best model --
    para_plot_group = [[0,1,2], [3], [4]]
    para_plot_color = [('g','r','k'),('k'),('k')]
        
    fig = plt.figure(figsize=(15, 5), dpi = 150)
    gs = GridSpec(1, len(para_plot_group) * 10, wspace = 0.2, bottom = 0.15, top = 0.85, left = 0.05, right = 0.95)

    for pp, (ppg, ppc) in enumerate(zip(para_plot_group, para_plot_color)):
        
        x = group_result['session_number']
        y = group_result['fitted_paras'][ppg,:]

        ax_grand = fig.add_subplot(gs[0, (pp*10) + 1])    
        ax_grand.set_prop_cycle(color = ppc)
        plt.axhline(0, c = 'k', ls = '--')
        ax_grand.plot(np.array([[-1] * len(ppg)]), np.array([group_result['fitted_paras_grand'][ppg]]), 's', markersize = 13, alpha = 0.8)
        
        ax = fig.add_subplot(gs[0, (pp*10+2) : (pp*10+10)], sharey = ax_grand)  
        ax.set_prop_cycle(color = ppc)
        
        plt.plot(x.T, y.T, linewidth = 0.7)
        
        for ny, yy in enumerate(y):
            ax.scatter(x[session_best_matched], yy[session_best_matched], s = marker_sizes, alpha = 0.9)
            ax.scatter(x[np.logical_not(session_best_matched)], yy[np.logical_not(session_best_matched)], 
                        edgecolor = ppc[ny], facecolors = 'none', s = marker_sizes, alpha = 0.9)
            sns.pointplot(data = yy.T, ax = ax_grand, color = ppc[ny], ci = 68, alpha = 0.8)

        plt.xlabel('Session number')
        if pp == 0: ax_grand.set_ylabel('Fitted parameters')
        plt.legend(fitted_para_names[ppg], bbox_to_anchor=(0,1.02,1,0.2), loc='lower center', ncol = 4)    
        plt.axhline(0, c = 'k', ls = '--')

        ax_grand.set_xticks([-1,0])
        ax_grand.set_xlim([-1.5,.5])
        ax_grand.set_xticklabels(['Overall','mean'], rotation = 45)
        plt.setp(ax.get_yticklabels(), visible=False)
    
    # plt.pause(10)      
    return

def plot_group_results(result_path = "..\\results\\model_comparison\\", group_results_name = 'group_results.npz',
                       average_session_number_range = None):
    #%%
    sns.set()
    
    # results_all_mice = pd.read_pickle(result_path + group_results_name)
    data = np.load(result_path + group_results_name, allow_pickle=True)
    group_results = data.f.group_results.item()
    results_all_mice = group_results['results_all_mice'] 
    
    if average_session_number_range is None:
        average_session_number_range = [0,np.inf]
        
    select_average_session = (average_session_number_range[0] <= results_all_mice['session_number']) & \
            (results_all_mice['session_number'] <= average_session_number_range[1])
            
    results_all_mice_session_filtered = results_all_mice[select_average_session]
                                            
    delta_AIC_para_notation = group_results['delta_AIC_para_notation']
    fitted_para_names = group_results['fitted_para_names']

    # == Plotting ==
    
    # -- 1. deltaAIC, aligned to session_number (Hattori Figure 1I) --
    fig = plt.figure(figsize=(10, 5), dpi = 150)
    ax = fig.subplots() 
    plt.axhline(0, c = 'k', ls = '--')
    palette = sns.color_palette(['orange', 'b', 'g', 'r'])
    # hh = sns.pointplot(data = results_all_mice.melt(id_vars = ['session_idx', 'session_number'], var_name='parameters',
    #                                            value_vars = delta_AIC_para_notation.tolist()),
    #               x = 'session_number', y = 'value' , hue = 'parameters', ci = 68, palette=palette)
    
    for pp, var in zip(palette, delta_AIC_para_notation.tolist()):
        means = results_all_mice.groupby('session_number')[var].mean()
        errs = results_all_mice.groupby('session_number')[var].sem()
        ax.errorbar(means.index, means, marker='o', yerr=errs, color = pp, label = var)
        
    ax.text(min(plt.xlim()),10,'(12) RW1972-noBias')
    ax.set_xlabel('Session number (actual)')
    ax.set_ylabel('$\\Delta$ AIC')
    # ax.set_title('%g mice' % group_results['n_mice'])
    plt.xticks(rotation=60, horizontalalignment='right')

    n_mice_per_session = results_all_mice.session_number.value_counts().sort_index()
    ax.plot(means.index, n_mice_per_session.values * max(plt.ylim()) / max(n_mice_per_session.values) * 0.9, 
            'ks-', label = 'max = %g mice' % group_results['n_mice'])

    if average_session_number_range is not None:
        patch = Rectangle((average_session_number_range[0], min(plt.ylim())), np.diff(np.array(average_session_number_range)), np.diff(np.array(plt.ylim())),
                          color = 'gray', linewidth = 1, fill= True, alpha = 0.2)
        ax.add_artist(patch)

    ax.legend()    

    # -- 2. deltaAIC, aligned to session_idx (Hattori Figure 1I) --
    fig = plt.figure(figsize=(10, 5), dpi = 150)
    ax = fig.subplots() 
    plt.axhline(0, c = 'k', ls = '--')
    palette = sns.color_palette(['orange', 'b', 'g', 'r'])
    # sns.pointplot(data = results_all_mice_session_filtered.melt(id_vars = ['session_idx', 'session_number'], var_name='parameters',
    #                                            value_vars = delta_AIC_para_notation.tolist()),
    #               x = 'session_idx', y = 'value' , hue = 'parameters', ci = 68, palette=palette)
        
    for pp, var in zip(palette, delta_AIC_para_notation.tolist()):
        means = results_all_mice_session_filtered.groupby('session_idx')[var].mean()
        errs = results_all_mice_session_filtered.groupby('session_idx')[var].sem()
        ax.errorbar(means.index, means, marker='o', yerr=errs, color = pp, label = var)

    ax.text(min(plt.xlim()),10,'(12) RW1972-noBias')
    ax.set_xlabel('Session Index')
    ax.set_ylabel('$\\Delta$ AIC')
    # ax.set_title('%g mice' % group_results['n_mice'])
    plt.xticks(rotation=60, horizontalalignment='right')

    n_mice_per_session = results_all_mice_session_filtered.session_idx.value_counts().sort_index()
    ax.plot(n_mice_per_session.values * max(plt.ylim()) / max(n_mice_per_session.values) * 0.9, 'ks-', label = 'max = %g mice' % group_results['n_mice'])

    ax.legend()   

    # -- 3. Prediction accuracy (Hattori Figure 1J)
    fig = plt.figure(figsize=(10, 5), dpi = 150)
    ax = fig.subplots() 
    plt.axhline(50, c = 'k', ls = '--')
    
    prediction_accuracies = results_all_mice_session_filtered[['mice','prediction_accuracy_NONCV', 'prediction_accuracy_bias_only']]
    prediction_accuracies = prediction_accuracies.rename(columns={"prediction_accuracy_NONCV": "Hattori 2019", "prediction_accuracy_bias_only": "Bias only"})
    prediction_accuracies = pd.DataFrame.melt(prediction_accuracies, id_vars = 'mice', var_name = 'models', value_name= 'value')
    prediction_accuracies.value *= 100

    x="mice"
    y="value"
    hue = 'models'
    hh = sns.violinplot(x=x, y=y, hue = hue, data = prediction_accuracies, inner="box")    
    
    ax.set_ylabel('Prediction accuracy % (nonCVed)')
    plt.xticks(rotation=45, horizontalalignment='right')
    ax.set_xlabel('')
    
    # Add Wilcoxon test
    box_pairs= [((s, "Hattori 2019"), (s, "Bias only")) for s in prediction_accuracies.mice.unique()]
    
    add_stat_annotation(hh, data=prediction_accuracies, x=x, y=y, hue=hue, box_pairs=box_pairs,
                        test='Wilcoxon', loc='inside', verbose=0, line_offset_to_box=0.2)
    
    # -- 4. fitted_para_names --
    # -- Separately
    for ff, fitted_para_this in enumerate(fitted_para_names):
        fig = plt.figure(figsize=(10, 5), dpi = 150)
        ax = fig.subplots() 
        plt.axhline(0, c = 'k', ls = '--')
        sns.violinplot(x="mice", y = fitted_para_this, data = results_all_mice_session_filtered, inner="box")    
        ax.set_title(fitted_para_this)
        plt.xticks(rotation=45, horizontalalignment='right')
        ax.set_xlabel('')
        ax.set_ylabel('')
        
    # -- Step sizes comparison
    fig = plt.figure(figsize=(10, 5), dpi = 150)
    ax = fig.subplots() 
    plt.axhline(0, c = 'k', ls = '--')
    
    step_sizes = results_all_mice_session_filtered[['mice','$\\alpha_{rew}$', ' $\\alpha_{unr}$', ' $\\delta$']]
    step_sizes = pd.DataFrame.melt(step_sizes, id_vars = 'mice', var_name = 'paras', value_name= 'value')
    x="mice"
    y="value"
    hue = 'paras'
    # sns.violinplot(x=x, y =y, hue=hue, data = step_sizes, inner="box")    
    hh = sns.boxplot(x=x, y =y, hue=hue, data = step_sizes, palette = sns.color_palette(['g', 'r', 'gray']))   
    hh.legend(bbox_to_anchor=(0,1.02,1,0.2), loc='lower center', ncol = 4)
    
    plt.xticks(rotation=45, horizontalalignment='right')
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Add Wilcoxon test
    box_pairs = []
    for s in step_sizes.mice.unique():
        box_pairs.extend([((s, '$\\alpha_{rew}$'), (s, ' $\\alpha_{unr}$')),
                          ((s, '$\\alpha_{rew}$'), (s, ' $\\delta$')),
                          ((s, ' $\\alpha_{unr}$'), (s, ' $\\delta$'))])
    
    add_stat_annotation(hh, data=step_sizes, x=x, y=y, hue=hue, box_pairs=box_pairs,
                        test='Wilcoxon', loc='inside', verbose=0, line_offset_to_box=0.2, line_offset = 0.03, line_height = 0.01, text_offset = 0.2)

    
    #%% -- 5. All mice statistics --
    fig = plt.figure(figsize=(10, 5), dpi = 150)
    gs = GridSpec(1, 6, wspace = 1, bottom = 0.15, top = 0.85, left = 0.1, right = 0.95)
    
    # 1> Prediction accuracy
    ax = fig.add_subplot(gs[0,0:2])  
    hh = sns.barplot(x = 'models', y = 'value', data = prediction_accuracies, order = ['Bias only', 'Hattori 2019'], 
                     capsize=.1, palette = sns.color_palette(['lightgray', 'darkviolet']))
    plt.axhline(50, c = 'k', ls = '--')
    
    add_stat_annotation(hh, data=prediction_accuracies, x = 'models', y = 'value', box_pairs=[['Bias only', 'Hattori 2019']],
                        test='Wilcoxon', loc='inside', verbose=0, line_offset_to_box=0)
    ax.set_xlabel('')
    ax.set_ylabel('Prediction accuracy % (not CV-ed)')
    ax.set_title('n = %g sessions'%len(prediction_accuracies))
    
    # 2> Step sizes 
    # Turn to mice-wise like Hattori??
    step_sizes = results_all_mice_session_filtered[['mice','$\\alpha_{rew}$', ' $\\alpha_{unr}$', ' $\\delta$']]
    step_sizes_mice_average = step_sizes.groupby(['mice']).mean()
    step_sizes_mice_average = pd.DataFrame.melt(step_sizes_mice_average)
    
    ax = fig.add_subplot(gs[0,2:4])  
    x = 'variable'
    y = 'value'
    hh = sns.boxplot(x = x, y = y, data = step_sizes_mice_average,  
                     palette = sns.color_palette(['g', 'r', 'gray']), width = 0.5)
    sns.swarmplot(x=x, y=y, data=step_sizes_mice_average, color = 'k')
    
    add_stat_annotation(hh, data = step_sizes_mice_average, x = x, y = y, 
                        box_pairs=[('$\\alpha_{rew}$', ' $\\alpha_{unr}$'), ('$\\alpha_{rew}$', ' $\\delta$'), (' $\\alpha_{unr}$', ' $\\delta$')],
                        test='Mann-Whitney', loc='inside', verbose=0, line_offset_to_box=0.1, line_offset = 0.03, line_height = 0.01, text_offset = 0.2)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('n = %g mice'% group_results['n_mice'])
    plt.axhline(0, c = 'k', ls = '--')
    
    # 3> beta(1/softmax_sigma)
    # Turn to mice-wise like Hattori??
    betas = results_all_mice_session_filtered[['mice', ' $\\sigma$', ' $b_L$']]
    betas_mice_average = betas.groupby(['mice']).mean()
    data = 1/betas_mice_average[' $\\sigma$']
    
    ax = fig.add_subplot(gs[0,4])  
    hh = sns.boxplot(data = data, width = 0.5, color = 'gray')
    sns.swarmplot(data = data, color = 'k')
    ax.set_ylim([0,16])
    ax.set_xticklabels(['$\\beta_{\\Delta Q}$'])
    ax.set_ylabel('')
    
    # 4> beta_0 (Hattori's bias)
    ax = fig.add_subplot(gs[0,5])  
    data = betas_mice_average[' $b_L$']
    hh = sns.boxplot(data = data, width = 0.5, color = 'gray')
    sns.swarmplot(data=data, color = 'k')
    
    plt.axhline(0, c = 'k', ls = '--')
    
    ax.set_xticklabels(['$\\beta_0$'])
    ax.set_ylabel('')
    ax.set_ylim([-2,2])

    
    plt.pause(5)
    #%%
    return
    



