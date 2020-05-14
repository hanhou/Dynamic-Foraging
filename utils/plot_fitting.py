# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:47:05 2020

@author: Han
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import pandas as pd

from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import cm


# import models.bandit_model_comparison
# import matplotlib as mpl 
# mpl.rcParams['figure.dpi'] = 300

# matplotlib.use('qt5agg')
plt.rcParams.update({'font.size': 14, 'figure.dpi': 150})

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_para_recovery(forager, true_paras, fitted_paras, para_names, para_bounds, para_scales, para_color_code, para_2ds, n_trials, fit_method):
    sns.reset_orig()
    n_paras, n_models = np.shape(fitted_paras)
    n_para_2ds = len(para_2ds)
    if para_scales is None: 
        para_scales = ['linear'] * n_paras
        
    # Color coded: 1 or the noise level (epsilon or softmax_temperature) 
    if para_color_code is None:
        if 'epsilon' in para_names:
            para_color_code = para_names.index('epsilon')
        elif 'softmax_temperature' in para_names:
            para_color_code = para_names.index('softmax_temperature')
        else:
            para_color_code = 1
            
    nn = min(4, n_paras + n_para_2ds)  # Column number
    mm = np.ceil((n_paras + n_para_2ds)/nn).astype(int)
    fig = plt.figure(figsize=(nn*4, mm*5), dpi = 100)
    
    fig.text(0.05,0.90,'Parameter Recovery: %s, Method: %s, N_trials = %g, N_runs = %g\nColor code: %s' % (forager, fit_method, n_trials, n_models, para_names[para_color_code]), fontsize = 15)

    gs = GridSpec(mm, nn, wspace=0.4, hspace=0.3, bottom=0.15, top=0.80, left=0.07, right=0.97) 
    
    xmin = np.min(true_paras[para_color_code,:])
    xmax = np.max(true_paras[para_color_code,:])
    if para_scales[para_color_code] == 'log':
        xmin = np.min(true_paras[para_color_code,:])
        xmax = np.max(true_paras[para_color_code,:])
        colors = cm.copper((np.log(true_paras[para_color_code,:])-np.log(xmin))/(np.log(xmax)-np.log(xmin)+1e-6)) # Use second as color (backward compatibility)
    else:
        colors = cm.copper((true_paras[para_color_code,:]-xmin)/(xmax-xmin+1e-6)) # Use second as color (backward compatibility)
      
    
    # 1. 1-D plot
    for pp in range(n_paras):
        ax = fig.add_subplot(gs[np.floor(pp/nn).astype(int), np.mod(pp,nn).astype(int)])
        plt.plot([para_bounds[0][pp], para_bounds[1][pp]], [para_bounds[0][pp], para_bounds[1][pp]],'k--',linewidth=1)
        
        # Raw data
        plt.scatter(true_paras[pp,:], fitted_paras[pp,:], marker = 'o', facecolors='none', s = 100, c = colors, alpha=0.7)
        
        if n_models > 1:   # Linear regression
            if para_scales[pp] == 'linear':
                x = true_paras[pp,:]
                y = fitted_paras[pp,:]
            else:  #  Use log10 if needed
                x = np.log10(true_paras[pp,:])
                y = np.log10(fitted_paras[pp,:])
                
            model = sm.OLS(y, sm.add_constant(x)).fit()
            b, k = model.params  
            r_square, p = (model.rsquared, model.pvalues)
            
            if para_scales[pp] == 'linear':
                plt.plot([min(x),max(x)], [k*min(x)+b, k*max(x)+b,], '-k', label = 'r^2 = %.2g\np = %.2g'%(r_square, p[1]))
            else:  #  Use log10 if needed
                plt.plot([10**min(x), 10**max(x)], [10**(k*min(x)+b), 10**(k*max(x)+b),], '-k', label = 'r^2 = %.2g\np = %.2g'%(r_square, p[1]))
        
        ax.set_xscale(para_scales[pp])
        ax.set_yscale(para_scales[pp])
        ax.legend()
        
        plt.title(para_names[pp])
        plt.xlabel('True')
        plt.ylabel('Fitted')
        plt.axis('square')
        
    # 2. 2-D plot
    
    for pp, para_2d in enumerate(para_2ds):
        
        ax = fig.add_subplot(gs[np.floor((pp+n_paras)/nn).astype(int), np.mod(pp+n_paras,nn).astype(int)])    
        legend_plotted = False
        
        # Connected true and fitted data
        for n in range(n_models):
            plt.plot(true_paras[para_2d[0],n], true_paras[para_2d[1],n],'ok', markersize=11, fillstyle='none', c = colors[n], label = 'True' if not legend_plotted else '',alpha=.7)
            plt.plot(fitted_paras[para_2d[0],n], fitted_paras[para_2d[1],n],'ok', markersize=7, c = colors[n], label = 'Fitted' if not legend_plotted else '',alpha=.7)
            legend_plotted = True
            
            plt.plot([true_paras[para_2d[0],n], fitted_paras[para_2d[0],n]], [true_paras[para_2d[1],n], fitted_paras[para_2d[1],n]],'-', linewidth=1, c = colors[n])
            
        # Draw the fitting bounds
        x1 = para_bounds[0][para_2d[0]]
        y1 = para_bounds[0][para_2d[1]]
        x2 = para_bounds[1][para_2d[0]]
        y2 = para_bounds[1][para_2d[1]]
        
        plt.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],'k--',linewidth=1)
        
        plt.xlabel(para_names[para_2d[0]])
        plt.ylabel(para_names[para_2d[1]])
        
        if para_scales[para_2d[0]] == 'linear' and para_scales[para_2d[1]] == 'linear':
            ax.set_aspect(1.0/ax.get_data_ratio())  # This is the correct way of setting square display
        
        ax.set_xscale(para_scales[para_2d[0]])
        ax.set_yscale(para_scales[para_2d[1]])

    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.show()


def plot_LL_surface(forager, LLsurfaces, CI_cutoff_LPTs, para_names, para_2ds, para_grids, para_scales, true_para, fitted_para, fit_history, fit_method, n_trials):
    
    sns.reset_orig()
            
    n_para_2ds = len(para_2ds)
    
    # ==== Figure setting ===
    nn_ax = min(3, n_para_2ds) # Column number
    mm_ax = np.ceil(n_para_2ds/nn_ax).astype(int)
    fig = plt.figure(figsize=(2.5+nn_ax*5, 1.5+mm_ax*5), dpi = 100)
    gs = GridSpec(mm_ax, nn_ax, wspace=0.2, hspace=0.35, bottom=0.1, top=0.84, left=0.07, right=0.97) 
    fig.text(0.05,0.88,'Likelihood Per Trial = p(data|paras, model)^(1/T): %s,\n Method: %s, N_trials = %g\n  True values: %s\nFitted values: %s' % (forager, fit_method, n_trials, 
                                                                                                                            np.round(true_para,3), np.round(fitted_para,3)),fontsize = 13)

    # ==== Plot each LL surface ===
    for ppp,(LLs, CI_cutoff_LPT, ps, para_2d) in enumerate(zip(LLsurfaces, CI_cutoff_LPTs, para_grids, para_2ds)):
    
        ax = fig.add_subplot(gs[np.floor(ppp/nn_ax).astype(int), np.mod(ppp,nn_ax).astype(int)]) 
        
        fitted_para_this = [fitted_para[para_2d[0]], fitted_para[para_2d[1]]]
        true_para_this = [true_para[para_2d[0]], true_para[para_2d[1]]]     
        para_names_this = [para_names[para_2d[0]], para_names[para_2d[1]]]
        para_scale = [para_scales[para_2d[0]], para_scales[para_2d[1]]]
                            
        for ii in range(2):
            if para_scale[ii] == 'log':
                ps[ii] = np.log10(ps[ii])
                fitted_para_this[ii] = np.log10(fitted_para_this[ii])
                true_para_this[ii] = np.log10(true_para_this[ii])
        
        dx = ps[0][1]-ps[0][0]
        dy = ps[1][1]-ps[1][0]
        extent=[ps[0].min()-dx/2, ps[0].max()+dx/2, ps[1].min()-dy/2, ps[1].max()+dy/2]

        # -- Gaussian filtering ---

        if dx > 0 and dy > 0:
            plt.imshow(LLs, cmap='plasma', extent=extent, interpolation='none', origin='lower')
            plt.colorbar()
        # plt.pcolor(pp1, pp2, LLs, cmap='RdBu', vmin=z_min, vmax=z_max)
        
        plt.contour(LLs, colors='grey', levels = 20, extent=extent, linewidths=0.7)
        # plt.contour(-np.log(-LLs), colors='grey', levels = 20, extent=extent, linewidths=0.7)
        
        # -- Cutoff LPT --
        plt.contour(LLs, levels = [CI_cutoff_LPT], colors = 'r', extent=extent)
        
        # ==== True value ==== 
        plt.plot(true_para_this[0], true_para_this[1],'ob', markersize = 20, markeredgewidth=3, fillstyle='none')
        
        # ==== Fitting history (may have many) ==== 
        if fit_history != []:
            
            # Compatible with one history (global optimizers) or multiple histories (local optimizers)
            for nn, hh in reversed(list(enumerate(fit_history))):  
                hh = np.array(hh)
                hh = hh[:,(para_2d[0], para_2d[1])]  # The user-defined 2-d subspace
                
                for ii in range(2):
                    if para_scale[ii] == 'log': hh[:,ii] = np.log10(hh[:,ii])
                
                sizes = 100 * np.linspace(0.1,1,np.shape(hh)[0])
                plt.scatter(hh[:,0], hh[:,1], s = sizes, c = 'k' if nn == 0 else None)
                plt.plot(hh[:,0], hh[:,1], '-' if nn == 0 else ':', color = 'k' if nn == 0 else None)
                
        # ==== Final fitted result ====
        plt.plot(fitted_para_this[0], fitted_para_this[1],'Xb', markersize=17)
    
        ax.set_aspect(1.0/ax.get_data_ratio()) 
        
        plt.xlabel(('log10 ' if para_scale[0] == 'log' else '') + para_names_this[0])
        plt.ylabel(('log10 ' if para_scale[1] == 'log' else '') + para_names_this[1])
        
    
    plt.show()
    
def plot_session_lightweight(fake_data, fitted_data = None, smooth_factor = 5):
    # sns.reset_orig()

    choice_history, reward_history, p_reward = fake_data
    
    # == Fetch data ==
    n_trials = np.shape(choice_history)[1]
    
    p_reward_fraction = p_reward[1,:] / (np.sum(p_reward, axis = 0))
                                      
    rewarded_trials = np.any(reward_history, axis = 0)
    unrewarded_trials = np.logical_not(rewarded_trials)
    
    # == Choice trace ==
    fig = plt.figure(figsize=(9, 4), dpi = 150)
        
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left = 0.1, right=0.8)

    # Rewarded trials
    ax.plot(np.nonzero(rewarded_trials)[0], 0.5 + (choice_history[0,rewarded_trials]-0.5) * 1.4, 
            'k|',color='black',markersize=20, markeredgewidth=2)

    # Unrewarded trials
    ax.plot(np.nonzero(unrewarded_trials)[0], 0.5 + (choice_history[0,unrewarded_trials] - 0.5) * 1.4, 
            '|',color='gray', markersize=10, markeredgewidth=1)
    
    # Base probability
    ax.plot(np.arange(0, n_trials), p_reward_fraction, color='y', label = 'base rew. prob.')
    
    # Smoothed choice history
    ax.plot(moving_average(choice_history, smooth_factor) , linewidth = 1.5, color='black', label = 'choice (smooth = %g)' % smooth_factor)
    
    # For each session, if any
    if fitted_data is not None:
        ax.plot(np.arange(0, n_trials), fitted_data[1,:], linewidth = 1.5, label = 'model') 

    ax.legend(fontsize = 10, loc=1, bbox_to_anchor=(0.985, 0.89), bbox_transform=plt.gcf().transFigure)
     
    ax.set_yticks([0,1])
    ax.set_yticklabels(['Left','Right'])
    # ax.set_xlim(0,300)
    
    # fig.tight_layout() 
    
    return ax
    
def plot_model_comparison_predictive_choice_prob(model_comparison, smooth_factor = 5):
    # sns.reset_orig()
    
    choice_history, reward_history, p_reward, trial_numbers = model_comparison.fit_choice_history, model_comparison.fit_reward_history, model_comparison.p_reward, model_comparison.trial_numbers
    if not hasattr(model_comparison,'plot_predictive'):
        model_comparison.plot_predictive = [1,2,3]
        
    n_trials = np.shape(choice_history)[1]

    ax = plot_session_lightweight([choice_history, reward_history, p_reward], smooth_factor = smooth_factor)
    # Predictive choice prob
    for bb in model_comparison.plot_predictive:
        bb = bb - 1
        if bb < len(model_comparison.results):
            this_id = model_comparison.results_sort.index[bb] - 1
            this_choice_prob = model_comparison.results_raw[this_id].predictive_choice_prob
            this_result = model_comparison.results_sort.iloc[bb]
           
            ax.plot(np.arange(0, n_trials), this_choice_prob[1,:] , linewidth = max(1.5-0.3*bb,0.2), 
                    label = 'Model %g: %s, Km = %g\n%s\n%s' % (bb+1, this_result.model, this_result.Km, 
                                                                                        this_result.para_notation, this_result.para_fitted))
    
    # Plot session ends
    for sesson_end in np.cumsum(trial_numbers):
        plt.axvline(sesson_end, color='b', linestyle='--', linewidth = 2)

    ax.legend(fontsize = 10, loc=1, bbox_to_anchor=(0.985, 0.89), bbox_transform=plt.gcf().transFigure)
     
    # ax.set_xlim(0,300)
    
    # fig.tight_layout() 
    
    return

def plot_model_comparison_result(model_comparison):
    sns.set()
    
    results = model_comparison.results
    
    # Update notations
    para_notation_with_best_fit = []
    for i, row in results.iterrows():
        para_notation_with_best_fit.append('('+str(i)+') '+row.para_notation + '\n' + str(np.round(row.para_fitted,2)))
        
    results['para_notation_with_best_fit'] = para_notation_with_best_fit
        
    fig = plt.figure(figsize=(12, 8), dpi = 150)
    gs = GridSpec(1, 5, wspace = 0.1, bottom = 0.1, top = 0.85, left = 0.23, right = 0.95)
    
    
    # -- 1. LPT -- 
    ax = fig.add_subplot(gs[0, 0])
    s = sns.barplot(x = 'LPT', y = 'para_notation_with_best_fit', data = results, color = 'grey')
    s.set_xlim(min(0.5,np.min(np.min(model_comparison.results[['LPT_AIC','LPT_BIC']]))) - 0.005)
    plt.axvline(0.5, color='k', linestyle='--')
    s.set_ylabel('')
    s.set_xlabel('Likelihood per trial')

    # -- 2. AIC, BIC raw --
    ax = fig.add_subplot(gs[0, 1])
    df = pd.melt(results[['para_notation_with_best_fit','AIC','BIC']], 
                 id_vars = 'para_notation_with_best_fit', var_name = '', value_name= 'IC')
    s = sns.barplot(x = 'IC', y = 'para_notation_with_best_fit', hue = '', data = df)

    # Annotation
    x_max = max(plt.xlim())
    ylim = plt.ylim()
    best_AIC = np.where(results.best_model_AIC)[0][0]
    plt.plot(x_max, best_AIC - 0.2, '*', markersize = 15)
    best_BIC = np.where(results.best_model_BIC)[0][0]
    plt.plot(x_max, best_BIC + 0.2, '*', markersize = 15)
    plt.ylim(ylim)
    
    s.set_yticklabels('')
    s.legend(bbox_to_anchor=(0,1.02,1,0.2), loc='lower left', ncol = 1)
    s.set_ylabel('')
    s.set_xlabel('AIC or BIC')

    # -- 3. log10_BayesFactor --
    ax = fig.add_subplot(gs[0, 2])
    df = pd.melt(results[['para_notation_with_best_fit','log10_BF_AIC','log10_BF_BIC']], 
                 id_vars = 'para_notation_with_best_fit', var_name = '', value_name= 'log10 (Bayes factor)')
    s = sns.barplot(x = 'log10 (Bayes factor)', y = 'para_notation_with_best_fit', hue = '', data = df)
    h_d = plt.axvline(-2, color='r', linestyle='--', label = 'decisive')
    s.legend(handles = [h_d,], bbox_to_anchor=(0,1.02,1,0.2), loc='lower left')
    # s.invert_xaxis()
    s.set_xlabel('log$_{10}\\frac{p(model)}{p(best\,model)}$')
    s.set_ylabel('')
    s.set_yticklabels('')
    
    # -- 4. Model weight --
    ax = fig.add_subplot(gs[0, 3])
    df = pd.melt(results[['para_notation_with_best_fit','model_weight_AIC','model_weight_BIC']], 
                 id_vars = 'para_notation_with_best_fit', var_name = '', value_name= 'Model weight')
    s = sns.barplot(x = 'Model weight', y = 'para_notation_with_best_fit', hue = '', data = df)
    ax.legend_.remove()
    plt.xlim([0,1.05])
    plt.axvline(1, color='k', linestyle='--')
    s.set_ylabel('')
    s.set_yticklabels('')
    
    # -- 5. Prediction accuracy --
    
    prediction_accuracy_NONCV = np.zeros(len(results))
    
    for rr, raw in enumerate(model_comparison.results_raw):
        this_predictive_choice_prob = raw.predictive_choice_prob
        this_predictive_choice = np.argmax(this_predictive_choice_prob, axis = 0)
        prediction_accuracy_NONCV[rr] = np.sum(this_predictive_choice == model_comparison.fit_choice_history) / model_comparison.n_trials
        
    results['prediction_accuracy_NONCV'] = prediction_accuracy_NONCV * 100

    ax = fig.add_subplot(gs[0, 4])
    s = sns.barplot(x = 'prediction_accuracy_NONCV', y = 'para_notation_with_best_fit', data = results, color = 'grey')
    plt.axvline(50, color='k', linestyle='--')
    ax.set_xlim(min(50,np.min(np.min(model_comparison.results[['prediction_accuracy_NONCV']]))) - 5)
    ax.set_ylabel('')
    ax.set_xlabel('Prediction_accuracy_NONCV')
    s.set_yticklabels('')

    return

def plot_confusion_matrix(confusion_results, order = None):
    sns.set()
    
    n_runs = np.sum(~np.isnan(confusion_results['raw_AIC'][0,0,:]))
    n_trials = confusion_results['n_trials']
    
    # === Get notations ==
    model_notations = confusion_results['models_notations']
    
    # Reorder if needed
    if order is not None:
        model_notations = ['('+str(ii+1)+') '+model_notations[imodel] for ii,imodel in enumerate(order)]
    else:
        model_notations = ['('+str(ii+1)+') '+ mm for ii,mm in enumerate(model_notations)]
        
    # === Plotting ==
    contents = [
               [['confusion_best_model_AIC','inversion_best_model_AIC'],
               ['confusion_best_model_BIC','inversion_best_model_BIC']],
               [['confusion_log10_BF_AIC','confusion_AIC'],
               ['confusion_log10_BF_BIC','confusion_BIC']],
               ]

    for cc, content in enumerate(contents):    
        fig = plt.figure(figsize=(10, 8.5))
        fig.text(0.05,0.97,'Model Recovery: n_trials = %g, n_runs = %g, True →, Fitted ↓' % (n_trials, n_runs))

        gs = GridSpec(2, 2, wspace=0.15, hspace=0.1, bottom=0.02, top=0.8, left=0.15, right=0.97)
    
        for ii in range(2):
            for jj in range(2):
                
                # Get data
                data = confusion_results[content[ii][jj]]
                
                # Reorder
                if order is not None:
                    data = data[:, np.array(order) - 1]
                    data = data[np.array(order) - 1, :]
                    
                # -- Plot --
                ax = fig.add_subplot(gs[ii, jj])
                
                # I transpose the data here so that the columns are ground truth and the rows are fitted results,
                # which is better aligned to the model comparison plot and the model-comparison-as-a-function-of-session in the real data.
                if cc == 0:
                    sns.heatmap(data.T, annot = True, fmt=".2g", ax = ax, square = True, annot_kws={"size": 10})
                else:
                    if jj == 0:
                        sns.heatmap(data.T, annot = True, fmt=".2g", ax = ax, square = True, annot_kws={"size": 10}, vmin=-2, vmax=0)
                    else:
                        sns.heatmap(data.T, annot = False, fmt=".2g", ax = ax, square = True, annot_kws={"size": 10})
                        
                set_label(ax, ii,jj, model_notations)
                plt.title(content[ii][jj])
        
        fig.show()
        

def set_label(h,ii,jj, model_notations):
    
    if jj == 0:
        h.set_yticklabels(model_notations, rotation = 0)
    else:
        h.set_yticklabels('')
        
    if ii == 0:
        h.set_xticklabels(model_notations, rotation = 45, ha = 'left')
        h.xaxis.tick_top()
    else:
        h.set_xticklabels('')
             











