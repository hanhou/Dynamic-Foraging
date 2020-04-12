# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:47:05 2020

@author: Han
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import cm

# matplotlib.use('qt5agg')
plt.rcParams.update({'font.size': 14})


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_para_recovery(forager, true_paras, fitted_paras, para_names, para_bounds, para_scales, para_color_code, para_2ds, n_trials, fit_method):
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
    fig = plt.figure(figsize=(nn*4, mm*5))
    
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


def plot_LL_surface(forager, LLsurfaces, para_names, para_2ds, para_grids, para_scales, true_para, fitted_para, fit_history, fit_method, n_trials):
    n_para_2ds = len(para_2ds)
    
    # ==== Figure setting ===
    nn_ax = min(3, n_para_2ds) # Column number
    mm_ax = np.ceil(n_para_2ds/nn_ax).astype(int)
    fig = plt.figure(figsize=(2.5+nn_ax*5, 1.5+mm_ax*5))
    gs = GridSpec(mm_ax, nn_ax, wspace=0.2, hspace=0.35, bottom=0.1, top=0.84, left=0.07, right=0.97) 
    fig.text(0.05,0.88,'Likelihood Per Trial = p(data|paras, model)^(1/T): %s,\n Method: %s, N_trials = %g\n  True values: %s\nFitted values: %s' % (forager, fit_method, n_trials, 
                                                                                                                            np.round(true_para,3), np.round(fitted_para,3)),fontsize = 13)

    # ==== Plot each LL surface ===
    for ppp,(LLs, ps, para_2d) in enumerate(zip(LLsurfaces, para_grids, para_2ds)):
    
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
        
        if dx > 0 and dy > 0:
            plt.imshow(LLs, cmap='plasma', extent=extent, interpolation='none', origin='lower')
            plt.colorbar()
        # plt.pcolor(pp1, pp2, LLs, cmap='RdBu', vmin=z_min, vmax=z_max)
        
        plt.contour(LLs, colors='grey', levels = 20, extent=extent, linewidths=0.7)
        # plt.contour(-np.log(-LLs), colors='grey', levels = 20, extent=extent, linewidths=0.7)
        
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
    
    
def plot_predictive_choice_prob(model_comparison):

    p_reward = model_comparison.data[2]
    choice_history = model_comparison.fit_choice_history
    reward_history = model_comparison.fit_reward_history
    
    smooth_factor = 5
    
    # == Fetch data ==
    n_trials = np.shape(choice_history)[1]
    
    p_reward_fraction = p_reward[1,:] / (np.sum(p_reward, axis = 0))
                                      
    rewarded_trials = np.any(reward_history, axis = 0)
    unrewarded_trials = np.logical_not(rewarded_trials)
    
    # == Choice trace ==
    fig = plt.figure(figsize=(9, 4))
        
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
    ax.plot(moving_average(choice_history, smooth_factor) , linewidth = 2, color='black', label = 'choice (smooth = %g)' % smooth_factor)
    
    # Predictive choice prob
    for bb in model_comparison.plot_predictive:
        if bb < len(model_comparison.results):
            this_id = model_comparison.results.index[bb]
            this_choice_prob = model_comparison.results_raw[this_id].predictive_choice_prob
            this_result = model_comparison.results.iloc[bb]
            
            ax.plot(this_choice_prob[1,:] , label = 'Model %g: %s, Km = %g\n%s\n%s' % (bb, this_result.model, this_result.Km, 
                                                                                       this_result.para_notation, this_result.para_fitted))
        
    ax.legend(fontsize = 10, loc=1, bbox_to_anchor=(0.985, 0.89), bbox_transform=plt.gcf().transFigure)
     
    ax.set_yticks([0,1])
    ax.set_yticklabels(['Left','Right'])
    ax.set_xlim(0,300)
    
    # fig.tight_layout() 
    
    return

