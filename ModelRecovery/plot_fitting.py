# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:47:05 2020

@author: Han
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import cm

# matplotlib.use('qt5agg')
plt.rcParams.update({'font.size': 15})


def plot_para_recovery(forager, true_paras, fitted_paras, para_names, para_bounds, para_scales, n_trials, fit_method, n_x0s):
    n_paras, n_models = np.shape(fitted_paras)
    if para_scales is None: 
        para_scales = ['linear'] * n_paras
        
    fig = plt.figure(figsize=((n_paras+1)*5, 5*1))
    
    if fit_method != 'DE':
        fit_method = fit_method + ' (n_x0s = %g)'%n_x0s
        
    fig.text(0.05,0.94,'Parameter Recovery: %s, Method: %s' % (forager, fit_method), fontsize = 15)

    gs = GridSpec(1,n_paras+1, wspace=0.3, hspace=0.5, bottom=0.13) 
    
    xmin = np.min(true_paras[1,:])
    xmax = np.max(true_paras[1,:])
    if para_scales[1] == 'log':
        xmin = np.min(true_paras[1,:])
        xmax = np.max(true_paras[1,:])
        colors = cm.copper((np.log(true_paras[1,:])-np.log(xmin)+1e-6)/(np.log(xmax)-np.log(xmin)+1e-6)) # Use second as color (backward compatibility)
    else:
        colors = cm.copper((true_paras[1,:]-xmin+1e-6)/(xmax-xmin+1e-6)) # Use second as color (backward compatibility)
      
    
    # 1. 1-D plot
    for pp in range(n_paras):
        ax = fig.add_subplot(gs[0,pp])
        plt.scatter(true_paras[pp,:], fitted_paras[pp,:], marker = 'o', facecolors='none', s = 100, c = colors, alpha=0.9)
        plt.plot([para_bounds[0][pp], para_bounds[1][pp]], [para_bounds[0][pp], para_bounds[1][pp]],'k--',linewidth=1)
        
        ax.set_xscale(para_scales[pp])
        ax.set_yscale(para_scales[pp])
        
        plt.title(para_names[pp])
        plt.xlabel('True')
        plt.ylabel('Fitted')
        plt.axis('square')
        
    # 2. 2-D plot
    ax = fig.add_subplot(gs[0,pp+1])    
    legend_plotted = False
    for n in range(n_models):
        
        plt.plot(true_paras[0,n], true_paras[1,n],'ok', markersize=12, fillstyle='none', c = colors[n], label = 'True' if not legend_plotted else '')
        plt.plot(fitted_paras[0,n], fitted_paras[1,n],'ok', markersize=8, c = colors[n], label = 'Fitted' if not legend_plotted else '')
        legend_plotted = True
        
        plt.plot([true_paras[0,n], fitted_paras[0,n]], [true_paras[1,n], fitted_paras[1,n]],'-', linewidth=1, c = colors[n])
        
        # Draw the fitting bounds
        x1, y1 = para_bounds[0]
        x2, y2 = para_bounds[1]
        
        plt.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],'k--',linewidth=1)
        
        plt.xlabel(para_names[0])
        plt.ylabel(para_names[1])
        ax.set_aspect(1.0/ax.get_data_ratio())  # This is the correct way of setting square display
        plt.title('n_trials = %g'%n_trials)
        
        ax.set_xscale(para_scales[0])
        ax.set_yscale(para_scales[1])

    
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.show()


def plot_LL_surface(LLs,fitted_para, true_para, fit_history, para_names, para_scales, ps, fit_method, n_x0s):
    
    if fit_method != 'DE':
        fit_method = fit_method + ' (n_x0s = %g)'%n_x0s

    # ==== Plot LL surface ===
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1) 
    
    # plt.pcolor(pp1, pp2, LLs, cmap='RdBu', vmin=z_min, vmax=z_max)
    
    for ii in range(2):
        if para_scales[ii] == 'log':
            ps[ii] = np.log10(ps[ii])
            fitted_para[ii] = np.log10(fitted_para[ii])
            true_para[ii] = np.log10(true_para[ii])
    
    dx = ps[0][1]-ps[0][0]
    dy = ps[1][1]-ps[1][0]
    extent=[ps[0].min()-dx/2, ps[0].max()+dx/2, ps[1].min()-dy/2, ps[1].max()+dy/2]
    
    plt.imshow(LLs, cmap='plasma', extent=extent, interpolation='none', origin='lower')
    plt.colorbar()
    
    plt.contour(-np.log(-LLs), colors='grey', levels = 20, extent=extent, linewidths=0.7)
    
    # ==== True value ==== 
    plt.plot(true_para[0], true_para[1],'ok', markersize = 20, markeredgewidth=3, fillstyle='none')
    
    # ==== Fitting history (may have many) ==== 
    if fit_history != []:
        
        # Compatible with one history (global optimizers) or multiple histories (local optimizers)
        for nn, hh in reversed(list(enumerate(fit_history))):  
            hh = np.array(hh)
            
            for ii in range(2):
                if para_scales[ii] == 'log': hh[:,ii] = np.log10(hh[:,ii])
            
            sizes = 100 * np.linspace(0.1,1,np.shape(hh)[0])
            plt.scatter(hh[:,0], hh[:,1], s = sizes, c = 'k' if nn == 0 else None)
            plt.plot(hh[:,0], hh[:,1], '-' if nn == 0 else ':', color = 'k' if nn == 0 else None)
            
    # ==== Final fitted result ====
    plt.plot(fitted_para[0], fitted_para[1],'Xk', markersize=17)

    ax.set_aspect(1.0/ax.get_data_ratio()) 
    plt.xlabel(('log10' if para_scales[0] == 'log' else ' ') + para_names[0])
    plt.ylabel(('log10' if para_scales[1] == 'log' else ' ') + para_names[1])
    plt.title('Log Likelihood p(data|parameter), method: %s'%fit_method)
    
    plt.show()
    