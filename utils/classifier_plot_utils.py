# -*- coding: utf-8 -*-
"""
Plotting functions for classifier, including 
    trajectories of predicted class probabiltiies alongside y pos/speed,
    and confusion matrices / feature importances for testing / analyzing model
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import os
import seaborn as sns
from utils.time_utils import frame_from_local_time, local_time_from_frame
from sklearn.tree import export_graphviz


def set_font_sizes(SMALL_SIZE=14, MEDIUM_SIZE=16, LARGE_SIZE=20):
    '''
    Sets font size for matplotlib
    From: https://stackoverflow.com/a/39566040
    '''
    font = {'family':'sans-serif',
            'sans-serif':['Arial'],
            'size': SMALL_SIZE}
    plt.rc('font', **font)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=LARGE_SIZE)   # fontsize of the figure title

state_colors = {
    'Alive': 'blue',
    'pre': 'gold',
    'dur': 'red',
    'post': 'black'
}

def plot_traj_ypos_spd(t, ypos, spd, spacing=100):
    '''
    Plots y position and speed as two subplots stacked vertically,
    with optional subsampling (spacing)
    '''
    # plot ypos
    ax1 = plt.subplot(311)
    ax1.plot(t[::spacing], ypos[::spacing], color='gray')
    ax1.set_ylabel('ypos')
    ax1.set_xlim(t[0], t[-1])
    ax1.set_ylim(-0.01, 1.01)
    
    # plot speed
    ax2 = plt.subplot(312, sharex=ax1)
    ax2.plot(t[::spacing], spd[::spacing], color='gray')
    ax2.set_ylabel('speed')
    ax2.set_ylim(-0.01, 50.01)
    
    return ax1, ax2

def plot_classifier_traj(t, ypos, spd, 
                         pred_times, pred_probs,
                         dic_class_to_i, dic_i_to_class,
                         spacing=100):
    '''
    Plots y position and speed as first two subplots,
    then plots evolution of classifier class probabilities as third subplot
    '''
    ax1, ax2 = plot_traj_ypos_spd(t, ypos, spd, spacing)
    
    # plot class probabilities
    ax3 = plt.subplot(313, sharex=ax1)
    for ic in [dic_class_to_i[x] for x in ['Alive', 'pre', 'dur', 'post']]:
        ax3.plot(pred_times, pred_probs[:, ic], 
                 color=state_colors[dic_i_to_class[ic]], 
                 label='P({})'.format(dic_i_to_class[ic]))
    ax3.set_ylim(-0.03, 1.03)
    ax3.set_ylabel('P(class)')
    ax3.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    
    return ax1, ax2, ax3
    
# colors to match CNE color style
ROI_state_colors = {
    'Alive': 'blue',
    'Cadaver': 'red',
    'NI': 'gray'
}


def plot_confusion(y_pred, y_true, labs):
    '''
    Given predicted y (y_pred) and true y (y_true) identities,
    with desired label ordering in labs, builds a confusion matrix
    and plots it, with appropriate normalizations to read out 
    precision (TP / (TP + FP)) and recall (TP / (TP + FN))
    '''

    # build the confusion matrix
    cm = confusion_matrix(y_pred, y_true, labels=labs)
    cm_df = pd.DataFrame(cm, index=labs, columns=labs)
    cm_fractrue_df = pd.DataFrame(cm / cm.sum(axis=0),
                              index=labs, columns=labs)
    cm_fracpred_df = pd.DataFrame((cm.T / cm.sum(axis=1)).T,
                              index=labs, columns=labs)
    
    # plot counts heatmap
    plt.subplot(131)
    plt.title('Counts')
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='viridis')
    plt.xlabel('true label')
    plt.ylabel('predicted label')

    # plot precision heatmap
    plt.subplot(132)
    plt.title('Proportion of predicted labels (precision)')
    sns.heatmap(cm_fracpred_df, annot=True, vmin=0, vmax=1, fmt='.2f', cmap='viridis')
    plt.xlabel('true label')
    plt.ylabel('predicted label')

    # plot recall heatmap
    plt.subplot(133)
    plt.title('Proportion of true labels (recall)')
    sns.heatmap(cm_fractrue_df, annot=True, vmin=0, vmax=1, fmt='.2f', cmap='viridis')
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    
    return cm_df


pred_legend = [Line2D([0], [0], color=state_colors['Alive']),
                Line2D([0], [0], color=state_colors['pre']),
                Line2D([0], [0], color=state_colors['dur']),
                Line2D([0], [0], color=state_colors['post'])]


def add_hr_x_axis(ax, t_df, zpos=36, hr_delta=2):
    '''
    Utility function for adding local time beneath an x axis
    '''
    start_time = t_df['start_time'][0]
    hr_max = np.floor(local_time_from_frame(ax.get_xlim()[1], start_time))
    hr_labels = np.ceil(np.arange(start_time, hr_max, hr_delta)).astype(int)
    hr_label_times = ['{}:00'.format(t % 24) for t in hr_labels]
    hr_locs = [frame_from_local_time(x, start_time) for x in hr_labels]
    ax2 = ax.twiny()#
    ax2.set_xticks(hr_locs)
    ax2.set_xticklabels(hr_label_times)
    ax2.xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
    ax2.xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom
    ax2.spines['bottom'].set_position(('outward', zpos))
    ax2.set_xlabel('Local Time (hr)')
    ax2.set_xlim(ax.get_xlim())
    return ax2


def plot_traj(roi, t_df, t_ed, npl=2, frame_sets=[], frame_cols=[],
                    frame_sets_bottom=[], frame_cols_bottom=[], title='',
                    ypos_color='k', spd_color='k', frame_alph=0.5, frame_lw=1, frame_zorder=1,
                    show_start_end=True, end_frame=None,
                    speed_mult=1):
    '''
    For a desired ROI, and using t_df, t_ed from a data directory, 
    plots trajectory of y position and speed.
    Can optionally plot lines on desired frames
        (frame_sets = [[f1,f2,f3], [g1,g2,g3]]) with 
        separate colors (frame_cols = ['fuchsia', 'green'])
    and can add custom title (appended to 'ROI #: {state}')
    '''

    which_i = roi - 1 # because ROIs are labeled 1, ..., 128
    state = t_df.Res.iloc[which_i] # cadaver, alive, or NI
    tau_frame = t_df['Best_tau'][which_i] # mark best tau if cadaver
    last_mvmt_frame = np.nan
    if state != 'Alive':
        last_mvmt_frame = t_df['Last_mvmt'][which_i] # mark last movement
    
    plt.suptitle('ROI {}: {} '.format(which_i+1, state)+title)

    # plot ypos
    ax0 = plt.subplot(npl, 1, 1)
    ax0.plot(t_ed['ypos'][:end_frame, which_i], color=ypos_color, alpha=0.8, zorder=5)

    # optionally mark individual frames (for example, to illustrate
    # which frames used in data processing for ML tools)
    for i in range(len(frame_sets)):
        frames = frame_sets[i]
        for fr in frames:
            if not np.isnan(fr):
                ax0.axvline(fr, color=frame_cols[i], alpha=frame_alph, linewidth=frame_lw, zorder=frame_zorder)
    
    ax0.set_ylabel('ypos')

    # plot speed
    ax1 = plt.subplot(npl, 1, 2)
    ax1.plot(t_ed['speed'][:end_frame, which_i] * speed_mult, color=spd_color, alpha=0.8, zorder=5)
    
    for i in range(len(frame_sets_bottom)):
        frames = frame_sets_bottom[i]
        for fr in frames:
            if not np.isnan(fr):
                ax1.axvline(fr, color=frame_cols_bottom[i], alpha=frame_alph, linewidth=frame_lw, zorder=frame_zorder)
    
    if show_start_end:
        ax0.axvline(tau_frame, color='black')#, label='tau')
        ax0.axvline(last_mvmt_frame, color='black')#, label='last movement')
        ax1.axvline(tau_frame, color='black')#, label='tau')
        ax1.axvline(last_mvmt_frame, color='black')#, label='last movement')
    ax1.set_ylabel('speed')
    ax1.set_xlabel('Frames since start')

    return ax0, ax1#, ax2



def plot_feature_vec(sub_df, state, gs, plotnum, params,
        ntoplot=200, alph=3,
        min_ht = -0.04, max_ht = 1.04, 
        min_spd = -25, max_spd = 1,
        min_t = 27, max_t = 39):
    '''
    Given an array of parameter values in training/validation/test data, 
    and a classifier state, plots feature vectors, splitting up 
    time / ypos / speed parameters into separate plots.
    Used in training/data_processing_overview.ipynb    
    '''

    pn = params['NPREV']; cn = params['NCURR']
    which_to_plot = np.random.randint(0, len(sub_df), ntoplot)
    alph /= ntoplot
    
    ax1 = plt.subplot(gs[plotnum])
    plt.title('{} time'.format(state))
    ax1.hist(sub_df[which_to_plot, 0], bins=40, orientation='horizontal', 
             color=state_colors[state])
    ax1.set_ylim(min_t, max_t)
    plt.xticks([], []) 
    
    
    ax2 = plt.subplot(gs[plotnum+1])
    plt.title('{} ypos'.format(state))
    ax2.set_ylim(min_ht, max_ht)
    ax2.plot(1+np.arange(pn), sub_df[which_to_plot, 1:(pn+1)].T, 
             'o', markeredgewidth=0, alpha=alph, color=state_colors[state])
    ax2.plot(1+pn+np.arange(cn), sub_df[which_to_plot, (pn+1):(1+pn+cn)].T, 
             'o', markeredgewidth=0, alpha=alph, color=state_colors[state])
    ax2.axvline(1+pn, linestyle='--', color='gray')
    fig_ht = ax2.get_ylim()[1] - ax2.get_ylim()[0]
    ax2.text(pn/2, min_ht-0.02*fig_ht, 'prev',
             horizontalalignment='center', verticalalignment='top')
    ax2.text(1+pn+cn/2, min_ht-0.02*fig_ht, 'curr',
             horizontalalignment='center', verticalalignment='top')
    plt.xticks([], [])  
   
    ax3 = plt.subplot(gs[plotnum+2])
    plt.title('{} speed (log)'.format(state))
    ax3.set_ylim(min_spd, max_spd)
    dat1 = sub_df[which_to_plot, (pn+cn+1):(1+2*pn+cn)].T
    dat1[dat1 <= 0] = np.nan
    ax3.plot(1+np.arange(pn), np.log(dat1),
             'o', markeredgewidth=0, alpha=alph, color=state_colors[state])
    dat2 = sub_df[which_to_plot, (2*pn+cn+1):].T
    dat2[dat2 <= 0] = np.nan
    ax3.plot(1+pn+np.arange(cn), np.log(dat2), 
             'o', markeredgewidth=0, alpha=alph, color=state_colors[state])
    ax3.axvline(1+pn, linestyle='--', color='gray')
    fig_ht = ax2.get_ylim()[1] - ax2.get_ylim()[0]
    ax3.text(pn/2, min_spd-0.02*fig_ht, 'prev',
             horizontalalignment='center', verticalalignment='top')
    ax3.text(1+pn+cn/2, min_spd-0.02*fig_ht, 'curr',
             horizontalalignment='center', verticalalignment='top')
    plt.xticks([], [])    
    
    return ax1, ax2, ax3



def plot_tree(clf, params, out_filename):
    '''
    Simple plotting function for single tree, for instance one 
    trained in the random forest classifier
    '''
    tree_in_forest = clf.estimators_[0]
    export_graphviz(tree_in_forest,
                    out_file=out_filename+'.dot',
                    feature_names=np.arange(params['num_features']),
                    filled=True,
                    rounded=True)
    os.system('dot -Tpng {}.dot -o {}.png'.format(out_filename, out_filename))


def plot_feature_importances(featimp, params):
    '''
    Plots random forest feature importances, coloring features by their type
    '''
    # set up colors, legend
    feat_colors = np.array(['default_color']*params['num_features'])
    feat_colors[0] = 'black'
    feat_colors[params['i_ypos_prev']] = 'red'
    feat_colors[params['i_ypos_curr']] = 'orange'
    feat_colors[params['i_spd_prev']] = 'green'
    feat_colors[params['i_spd_curr']] = 'violet'

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='local time', markerfacecolor='black'),
                       Line2D([0], [0], marker='o', color='w', label='ypos prev', markerfacecolor='red'),
                       Line2D([0], [0], marker='o', color='w', label='ypos curr', markerfacecolor='orange'),
                       Line2D([0], [0], marker='o', color='w', label='speed prev', markerfacecolor='green'),
                       Line2D([0], [0], marker='o', color='w', label='speed curr', markerfacecolor='violet')]
    
    # plot
    #plt.figure()
    ax = plt.subplot(111)
    plt.scatter(np.arange(len(featimp)), featimp, color=feat_colors)
    ax.set_xlabel('Feature index')
    ax.set_ylabel('Feature importance')
    plt.legend(handles=legend_elements, title='Feature Type')
    #plt.show()