# -*- coding: utf-8 -*-
"""
When run inside an output folder from a classifier run,
this file plots ypos, speed for all ROIs 

Intended use case: navigate to a model output folder
(i.e. outputs/your_run_date_tag/), then run python plot_trajs.py
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

##########################################
############## INPUT FILE SETTINGS
##########################################

model_info_f = 'model_info.p' 
model_info_d = pickle.load( open( model_info_f , "rb" ) )
input_d = model_info_d['input_d']

import sys; sys.path.append(input_d['home_dir'])
from utils.time_utils import get_cur_time, get_day_thru_sec, \
        hr_min_sec_from_hr_float, get_exp_time_from_filename
from utils.load_data_utils import make_ed_from_bin_files
from utils.classifier_plot_utils import plot_classifier_traj

run_time = get_day_thru_sec()
run_dir = os.path.dirname(model_info_f)

predYprobs_f = os.path.join(run_dir, 'class_probs/predYprobs.p')
predYprobs = pickle.load( open( predYprobs_f, 'rb'))

classify_times = model_info_d['classify_times']

# load clf for mapping info
clf_filepath = model_info_d['clf_file']
clf = pickle.load(open(clf_filepath, 'rb'))
# know which indices correspond to classes
dic_class_to_i = dict((y,x) for (x,y) in enumerate(clf.classes_))
dic_i_to_class = dict((x,y) for (x,y) in enumerate(clf.classes_))

# experiment name for outputs
run_name = input_d['run_name']
# home directory (where outputs save)
home_dir = input_d['home_dir']
# data to read
centroid_bin_file = input_d['centroid_bin_file']
time_bin_file = input_d['time_bin_file']

###########################################
############## PROCESS SETTINGS
###########################################

# add timestamp

cur_time = get_cur_time()
expmt_start_time = get_exp_time_from_filename(centroid_bin_file) 



out_dir = os.path.join(run_dir,
       'manual_all_ROI_plots', 'plots_{}-{}-{}_{}-{}-{}'.format(*run_time))

# set up subdirectories for various outputs
#out_plot_dir = os.path.join(out_dir, 'out_plots')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    #os.makedirs(out_plot_dir)

# open this current script and save it to output directory
with open(os.path.join(out_dir, os.path.basename(__file__)), 'w') as f:
    f.write(open(__file__, "r").read())


# try accessing the binary files
print('Checking whether I can read the binary files:')
try:
    ed = make_ed_from_bin_files(centroid_bin_file, time_bin_file)
    print('Yep! Last time the time binary file was updated: '+\
              '{:.3f} ({:02d}:{:02d}:{:02d})\n'.format(ed['ts'][-1], *hr_min_sec_from_hr_float(ed['ts'][-1])))
except:
    raise IOError('No, cannot read the binary files.')
    
ROIs = 1+np.arange(128) 


# get centroid, time binary file info
ed = make_ed_from_bin_files(centroid_bin_file, time_bin_file)
ts = expmt_start_time + np.arange(ed['ypos'].shape[0])/3/60/60

# take last time frame
grab_index = -1
                               
# plot all ROIs
for i in np.arange(128):
    
    if i % 10 == 9:
        print('Plotting ROI {} of 128'.format(i+1))
  
    ypos = ed['ypos'][:grab_index, i] 
    spd = ed['speed'][:grab_index, i]
    
            
    plt.figure(figsize=(12,8))
    plt.suptitle('ROI {} at {:02d}:{:02d}:{:02d}'.format(i+1, *hr_min_sec_from_hr_float(cur_time)))
    plot_classifier_traj(t=ts[:grab_index], ypos=ypos, spd=spd,
                         pred_times=classify_times,
                         pred_probs=predYprobs[:, i, :],
                         dic_class_to_i=dic_class_to_i,
                         dic_i_to_class=dic_i_to_class)
    plot_name = os.path.join(out_dir, 
         'plot_time_{}_{}_{}_roi_{}.png'.format(
                 *hr_min_sec_from_hr_float(cur_time),
                 i+1))
    plt.xlabel('Local time')
    plt.savefig(plot_name, bbox_inches='tight', dpi=100)
    plt.close()
            
                