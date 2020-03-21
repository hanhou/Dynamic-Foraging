# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:47:05 2020

@author: Han
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import cm

def plot_para_recovery(forager, simulated_para, fitted_para, para_names, para_bounds):
    n_paras, n_models = np.shape(fitted_para)
    
    fig = plt.figure(figsize=((n_paras+1)*5, 5*1))
    
    fig.text(0.05,0.94,'Parameter Recovery: %s' % (forager), fontsize = 15)

    gs = GridSpec(1,n_paras+1, wspace=0.3, hspace=0.5, bottom=0.13) 
    
    colors = cm.binary(((simulated_para[1,:]-para_bounds[0][1]+1e-6)/(para_bounds[1][1]-para_bounds[0][1]+1e-6)+0.2)/1.2) # Use std as color
    
    for pp in range(n_paras):
        fig.add_subplot(gs[0,pp])
        plt.scatter(simulated_para[pp,:], fitted_para[pp,:], marker = 'o', facecolors='none', s = 100, c = colors)
        plt.plot([para_bounds[0][pp], para_bounds[1][pp]], [para_bounds[0][pp], para_bounds[1][pp]],'k--',linewidth=1)
        
        plt.title(para_names[pp])
        plt.xlabel('Simulated para')
        plt.ylabel('Fitted para')
        plt.axis('square')
        
    ax = fig.add_subplot(gs[0,pp+1])    
    for n in range(n_models):
        plt.plot(simulated_para[0,n], simulated_para[1,n],'ok', markersize=12, fillstyle='none', c = colors[n])
        plt.plot(fitted_para[0,n], fitted_para[1,n],'ok', markersize=8, c = colors[n])
        plt.plot([simulated_para[0,n], fitted_para[0,n]], [simulated_para[1,n], fitted_para[1,n]],'-', linewidth=1, c = colors[n])
        plt.xlabel(para_names[0])
        plt.ylabel(para_names[1])
        ax.set_aspect(1.0/ax.get_data_ratio())  # This is the correct way of setting square display
    
    plt.show()
