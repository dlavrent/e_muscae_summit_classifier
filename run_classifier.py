# -*- coding: utf-8 -*-
"""
Real-time E. muscae infection classifier:
    loads a classifier object (clf) previously 
        trained using multiple experiments (see training/ folder)
    extracts data from margo's outputted binary files,
    converts collected data into proper format for 
        classification,
    predicts class probabilities (during-, pre-, 
        post-summiting, or not infected ("alive"))
    optionally sends an email when an ROI is predicted to be
        during summiting
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import bottleneck as bn
import os
import pickle
from scipy.stats import rankdata

from utils.time_utils import get_cur_day, get_cur_time, get_day_thru_sec, \
        hr_min_sec_from_hr_float, get_exp_time_from_filename
from utils.email_utils import send_email
from utils.load_data_utils import make_ed_from_bin_files
from utils.process_data_utils import get_frames, make_X_from_frames_roi, \
    printf, ROI_pos_on_board
from utils.prediction_utils import longest_1_stretch_end
from utils.classifier_plot_utils import plot_classifier_traj

# IMPORT INPUT FILE HERE
from input_files.input_file_example import input_d


##########################################
############## INPUT FILE SETTINGS
##########################################

# experiment name for outputs
run_name = input_d['run_name']
# computer name (for emailing)
cpu_name = input_d['cpu_name']
# home directory (where outputs save)
home_dir = input_d['home_dir']
# classifier files
clf_name = input_d['clf_name']
params_name = input_d['params_name']
# data to read
centroid_bin_file = input_d['centroid_bin_file']
time_bin_file = input_d['time_bin_file']
# email settings
do_email = input_d['do_email']
email_recip = input_d['email_recip']
# do in real time?
real_time = input_d['real_time']
# adjust for no daylight savings time?
toZT = lambda x: x + input_d['NO_DST']
# identify non-summiters?
do_non_summiter_calling = input_d['do_non_summiter_calling']
n_extra_non_summiters = input_d['n_extra_non_summiters']
# save plots?
do_plot = input_d['do_plot']
# time settings
TIME_OFFSET0 = input_d['TIME_OFFSET0']
SLEEP_TIME = input_d['SLEEP_TIME']
# classifier time settings
classify_freq = input_d['classify_freq']
classify_start = input_d['classify_start']
classify_stop = input_d['classify_stop']
# classifier decision setting
Yprob_consec_frames = input_d['Yprob_consec_frames']
try:
    NROI = input_d['number_of_ROIs']
except:
    NROI = 128

###########################################
############## PROCESS SETTINGS
###########################################

# add timestamp
run_time = get_day_thru_sec()
run_time_str = '{}-{:02}-{:02}_{:02}-{:02}-{:02}'.format(*run_time)
out_dir = os.path.join(home_dir, 'outputs', run_name + '_' + run_time_str)

# set up subdirectories for various outputs
call_plot_dir = os.path.join(out_dir, 'emailed_plots')
if do_non_summiter_calling:
    non_summit_plot_dir = os.path.join(out_dir, 'additional_non_summit_plots')
class_prob_dir = os.path.join(out_dir, 'class_probs')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    os.makedirs(call_plot_dir)
    if do_non_summiter_calling:
        os.makedirs(non_summit_plot_dir)
    os.makedirs(class_prob_dir)
    
# save all printed statements
run_log = os.path.join(out_dir, 'run_log.txt')
# companion for predYprobs.p to track when it's updated
pred_log = os.path.join(class_prob_dir, 'pred_log.txt')
# store whether emails were sent
email_statuses = []
# save summiting ROIs and time they were called
out_table = []
out_table_cols = ['time', 'CPU', 'summiter ROI', '(row)-position', ''] 
if do_non_summiter_calling:
    out_table_cols += sum([['non-summiter ROI {}'.format(i+1), '(row)-position'] 
        for i in range(n_extra_non_summiters)], [])
out_table_path = os.path.join(out_dir, 
                              'call_table_{}_{}.csv'.format(cpu_name, run_time_str))
# open this current script and save it to output directory
with open(os.path.join(out_dir, os.path.basename(__file__)), 'w') as f:
    f.write(open(__file__, "r").read())
    
# open plotting script and save it to output directory
plotting_script = 'utils/plot_trajs_with_class_probs.py'
with open(os.path.join(out_dir, 'plot_trajs_with_class_probs.py'), 'w') as f:
    f.write(open(plotting_script, "r").read())


###########################################
############## MODEL
###########################################

# SPECIFY MODEL (previously trained)
model_dir = os.path.join(home_dir, 'models')
clf_filepath = os.path.join(model_dir, clf_name)
params_filepath = os.path.join(model_dir, params_name)
clf = pickle.load(open(clf_filepath, 'rb'))
params = pickle.load(open(params_filepath, 'rb'))

# know which indices correspond to classes
dic_class_to_i = dict((y,x) for (x,y) in enumerate(clf.classes_))
dic_i_to_class = dict((x,y) for (x,y) in enumerate(clf.classes_))


###########################################
############## DATA
###########################################

# set a label for the welcome email 
expmt_label = re.findall('\/([\w\d\-\_]+)\/raw_data\/', centroid_bin_file)[0]

# check the files exist!
if not os.path.exists(centroid_bin_file) or not os.path.exists(time_bin_file):
    raise IOError('Centroid or time binary file not found')
    
# extract relevant times
classify_times = np.arange(classify_start, classify_stop, classify_freq)
cur_time = get_cur_time(add_offset=TIME_OFFSET0)
expmt_start_time = get_exp_time_from_filename(centroid_bin_file)  
expmt_start_day = get_cur_day()

# try accessing the binary files
printf('Checking whether I can read the binary files:', run_log)
try:
    ed = make_ed_from_bin_files(centroid_bin_file, time_bin_file, NROI)
    printf('Yep! Last time the time binary file was updated: '+\
              '{:.3f} ({:02d}:{:02d}:{:02d})\n'.format(ed['ts'][-1], *hr_min_sec_from_hr_float(ed['ts'][-1])),
           run_log)
except:
    raise IOError('No, cannot read the binary files.')

ROIs = 1+np.arange(NROI) 



###########################################
############## CLASSIFIER SETTINGS
###########################################


# save model information
model_info = {}
model_info['input_d'] = input_d
model_info['params_file'] = params_filepath
model_info['clf_file'] = clf_filepath
model_info['script_start_time'] = run_time
model_info['centroid_file'] = centroid_bin_file
model_info['time_file'] = time_bin_file
model_info['classify_times'] = classify_times
model_info['offset'] = TIME_OFFSET0
pickle.dump(model_info, open(os.path.join(out_dir, 'model_info.p'), 'wb'))

    
# print welcome message
start_msg = '~~~ Real-time classifier ~~~\n'+\
      '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'+\
      'Current time: {:.3f} ({:02d}:{:02d}:{:02d})\n'.format(cur_time, 
                     *hr_min_sec_from_hr_float(cur_time))+\
      'Experiment: {}\n'.format(expmt_label)+\
      'CPU: {}\n'.format(cpu_name)+\
      'Classifier: {}\n'.format(clf_name)+\
      'Params: {}\n'.format(params_name)+\
      'Tracking {} ROIs\n'.format(NROI)+\
      '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'+\
      'Frames for classification will collect from\n'+\
      '\t{:.3f} ({:02d}:{:02d}:{:02d}) to\n'.format(classify_start, 
                     *hr_min_sec_from_hr_float(classify_start))+\
      '\t{:.3f} ({:02d}:{:02d}:{:02d}) every\n'.format(classify_stop, 
                     *hr_min_sec_from_hr_float(classify_stop))+\
       '\t{:03} minutes\n'.format(classify_freq*60)+\
      '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'+\
      'Outputs saving to:\n\t{}\n'.format(out_dir.replace('\\', '/'))+\
      '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
      
     
printf(start_msg, run_log)

if do_email:
    email_sent, email_status = send_email(email_recip, 
           subject='{}: Classifier launched at {:02d}:{:02d}:{:02d}!'.format(
                   cpu_name, *hr_min_sec_from_hr_float(cur_time)),
           content=start_msg)
    email_statuses.append([get_cur_time(add_offset=TIME_OFFSET0), email_status])
      
     
# initialize variables storing data

# array of X features (nFrames * nROI * nFeatures (specified by model))
allXes = np.zeros((len(classify_times), len(ROIs), params['num_features']))
# array of predicted class probabilities (nFrames * nROI * 4)
predYprobs = np.zeros((len(classify_times), len(ROIs), 4))
# list of ROIs that appear dead (full of NaN's)
nanROIs = np.full((len(ROIs),), False)
# total called summitters
n_total_called_summiters = 0; all_called_summiter_ROI_is = np.array([])
if do_non_summiter_calling:
    n_total_called_non_summiters = 0; all_called_non_summiter_ROI_is = np.array([])

# compare current time to the array of observation times.
# if you run this code before the set start time for observations,
# you will just wait until cur_time reaches the set start time.
# if you run it and the current time falls within the 
# observation time window (e.g. you stopped this code mid-experiment
# and are now running it again), then you will just hop to wherever
# in classify_times matches where current time is
cur_i = np.where(cur_time < classify_times)[0][0]  # "index" of cur_time 
                                                   # within classify_times

eligible_ROIs = np.arange(NROI)

for classify_i in range(cur_i, len(classify_times)):
    
    # get next classification time
    classify_time = classify_times[classify_i]
    
    # adjust time offset
    TIME_OFFSET = TIME_OFFSET0 + 24*(get_cur_day() != expmt_start_day)      
        
    # if running this script post data collection, then there's 
    # no need to wait for future data, so just take the data at 
    # the classification times
    cur_time = classify_time; ZT_cur_time = toZT(cur_time)
    # if doing in real time, then wait for next classification
    # by trying every SLEEP_TIME seconds
    if real_time:        
        TIME_OFFSET = TIME_OFFSET0 + 24*(get_cur_day() != expmt_start_day)
        cur_time = get_cur_time(add_offset=TIME_OFFSET); ZT_cur_time = toZT(cur_time)
        while cur_time < classify_time:
            printf('Current time: {:.3f} ({:02d}:{:02d}:{:02d})'.format(cur_time, 
                             *hr_min_sec_from_hr_float(cur_time)) +
                   ', next classification: {:02d}:{:02d}:{:02d}\n'.format(
                             *hr_min_sec_from_hr_float(classify_time)), run_log)
            time.sleep(SLEEP_TIME)
            TIME_OFFSET = TIME_OFFSET0 + 24*(get_cur_day() != expmt_start_day)
            cur_time = get_cur_time(add_offset=TIME_OFFSET); ZT_cur_time = toZT(cur_time)
         
    # we have now started classification
    if classify_i == 0:
        printf('Classification has begun!', run_log)  
        
    # internal time counter to track how long 
    # it takes to process current frame
    t0 = time.time()
    
    # make cur_time more readable
    cur_hr, cur_min, cur_sec = hr_min_sec_from_hr_float(cur_time)
    printf('\tProcessing ROIs at ({:02d}:{:02d}:{:02d}):'.format(
            *hr_min_sec_from_hr_float(cur_time)), run_log) 
    
    # load in data
    ed = make_ed_from_bin_files(centroid_bin_file, time_bin_file, NROI)
    #ts = ed['ts']
    ts = expmt_start_time + np.arange(ed['ypos'].shape[0])/3/60/60
    
    # check binary file has updated
    bin_file_last_time = ts[-1]
    
    # take position of trajectory data corresponding to current time
    grab_index = np.where(ts < cur_time)[0][-1]     

    # clock how long it takes to load data (should be ~3 seconds)
    t_load = time.time()
    printf('\tElapsed {:.3f} sec: data loaded, last binary file time {:.3f}'.format(
            t_load - t0, bin_file_last_time), run_log)

    # now go through each ROI
    for ir in range(len(ROIs)):   
        
        roi = ROIs[ir]
                    
        # only process relevant trajectories 
        # that aren't all NaN values 
        if not nanROIs[ir]:

            # get relevant frames
            prev_frames, curr_frames = get_frames(grab_index-1, params)
            grab_frames = np.concatenate((prev_frames, curr_frames))

            # extract y pos data
            ypos = ed['ypos'][:grab_index, roi-1] 

            # mark if ROI is all NaN's
            if np.all(np.isnan(ypos)):
                # don't bother extracting ypos, speed 
                # (taking the moving average takes time)
                #printf('ROI {} is useless'.format(ROIs[ir]), run_log)
                nanROIs[ir] = True
                allXes[classify_i, ir, :] = np.zeros(params['num_features'])          
            
            else: # otherwise, continue processing
                # extract speed data
                spd = ed['speed'][:grab_index, roi-1]
                
                # do mov_avg if that's what model was trained on
                if params['do_mov_avg']:
                    ypos = bn.move_mean(ypos, window=params['wl'], min_count=1)
                    spd = bn.move_mean(spd, window=params['wl'], min_count=1)

                # make X data, store it in allXes
                X_ex = make_X_from_frames_roi(grab_index, grab_frames, ypos, spd, ed)  
                allXes[classify_i, ir, :] = X_ex  
            
    t_features = time.time()
    printf('\tElapsed {:.3f} sec: ROIs converted to features and plotted'.format(t_features - t0), run_log)
    
    # now use classifier to predict class probabilities
    # of all non-NaN ROIs, store in predYprobs
    goodROIpos = np.arange(NROI)[~nanROIs]
    predYprobs[classify_i, goodROIpos] = clf.predict_proba(allXes[classify_i, goodROIpos, :])
    t_class = time.time()
    printf('\tElapsed {:.3f} sec: classification done'.format(t_class - t0), run_log)
    
    ############## MAY NEED TO MODIFY HERE!
    # find the frames in the predYprobs array in which P(dur) > P(alive)
    dur_over_alive = predYprobs[:(classify_i+1), :, dic_class_to_i['dur']] - predYprobs[:(classify_i+1), :, dic_class_to_i['Alive']]
    Yprob_condition = dur_over_alive > 0
    
    where_Yprobs_recorded = np.apply_over_axes(np.sum, predYprobs, [1,2])[:, 0, 0] > 0
    # now find the consecutive stretches in which P(dur) > P(alive)
    # and see if any are above the threshold stretch length of Yprob_consec_frames
    consec_dur_condition = np.apply_along_axis(longest_1_stretch_end, 0, Yprob_condition, Yprob_consec_frames)
    # any ROIs that have P(dur) > P(alive) >= desired consecutive stretch
    # are called in call_summitter_ROI_is:
    current_called_summiter_ROI_is = np.where(consec_dur_condition[-1])[0]
    # add to previously called ROIs
    all_called_summiter_ROI_is = np.concatenate((all_called_summiter_ROI_is, current_called_summiter_ROI_is)).astype(int)
    n_current_called_summiters = len(current_called_summiter_ROI_is)
    n_total_called_summiters += n_current_called_summiters
    eligible_ROIs = eligible_ROIs[~np.isin(eligible_ROIs, current_called_summiter_ROI_is)]
    
    # compute non-summiting score
    # first, identify region in which a classification probability recorded
    where_Yprobs_recorded = np.apply_over_axes(np.sum, predYprobs, [1,2])[:, 0, 0] > 0
    # then, extract the class probabilities over that time 
    posYprobs = predYprobs[where_Yprobs_recorded]
    # next, for each ROI, identify fraction of time that P(alive) > all others
    frac_alive_argmax = (np.argmax(posYprobs, axis=2) == dic_class_to_i['Alive']).sum(0) / posYprobs.shape[0]
    # average P(alive):
    average_p_alive = np.mean(posYprobs[:, :, dic_class_to_i['Alive']], axis=0)
    # then, for each ROI, find the max value of P(dur)
    max_dur_prob = np.max(posYprobs[:, :, dic_class_to_i['dur']], axis=0)
    # then, for each ROI, find fraction of time indices in which speed = 0
    frac_speeds_0 = np.sum(ed['speed'][:grab_index] == 0, axis=0) / ed['speed'].shape[0]
    # then, for each ROI, find its current speed (over past 'wl' time window)
    cur_speeds = np.mean(ed['speed'][-params['wl']:,:], 0)
    # finally, compute its percentile rank (0: speed=0, 1: fastest speed)
    cur_speed_pctiles = (rankdata(cur_speeds, 'min') -1)/127

    # putting it all together, this NEW non-summiter score is
    # (average P(alive)) * 
    # (1 - max[P(dur)]) * 
    # (fraction of speeds 0 < 0.9) *
    # (cur_speed_percentile)
    non_summiter_scores = average_p_alive*(1-max_dur_prob)*(frac_speeds_0 < 0.9)*(cur_speed_pctiles)
    argsorted_non_summiter_scores =  np.argsort(non_summiter_scores)[::-1]
    
    # save all non summiter scores
    non_sum_df = pd.DataFrame(np.vstack((ROIs, non_summiter_scores)).T, columns=['ROI', 'non_summit_score'])
    non_sum_df = non_sum_df.sort_values('non_summit_score', ascending=0)
    non_sum_df.to_csv(os.path.join(out_dir, 'non_summit_scores.csv'), index=False)
    
    summit_pngs = [] # store trajectory images
    non_summit_pngs = []
    t_call = time.time()
    printf('\tElapsed {:.3f} sec: {} ROIs called summiting'.format(t_call - t0, n_current_called_summiters), run_log)
    current_called_non_summiter_ROI_is = []
    
    # save predYprobs
    predYprobs_file = os.path.join(class_prob_dir, "predYprobs.p")
    pickle.dump(predYprobs, open(predYprobs_file, "wb" ))
    
    # for each summiter...                           
    for i in current_called_summiter_ROI_is:
        printf('\t\t!~!~!~!~! PREDICTED SUMMITING: ROI {}'.format(i+1), run_log) 
        time_str = '{:02}:{:02}:{:02}'.format(*hr_min_sec_from_hr_float(cur_time))
        out_line = [time_str, cpu_name, i+1, ROI_pos_on_board(i+1), '']
    
        if do_plot:
            # get y, spd for ROI
            ypos = ed['ypos'][:grab_index, i] 
            spd = ed['speed'][:grab_index, i]
            
            # plot summiter                
            plt.figure(figsize=(12,8))
            plt.suptitle('CPU: {}, ROI {} ({}), predicted summiting at {:02d}:{:02d}:{:02d}'.format(
                         cpu_name, 
                         i+1,
                         ROI_pos_on_board(i+1),
                         *hr_min_sec_from_hr_float(cur_time)))
            ax1, ax2, ax3 = plot_classifier_traj(t=ts[:grab_index], 
                     ypos=ypos, spd=spd,
                     pred_times=classify_times[:classify_i], 
                     pred_probs=predYprobs[:classify_i, i, :],
                     dic_class_to_i=dic_class_to_i, dic_i_to_class=dic_i_to_class)
            plot_name = os.path.join(call_plot_dir, 
                                     'time_{}_{}_{}_roi_{}.png'.format(
                                             *hr_min_sec_from_hr_float(cur_time),
                                             i+1))
            trange = np.arange(np.round(ts[0]), ts[grab_index]+0.001, 1.0)
            ax3.set_xticks(trange)
            ax3.set_xlabel('local time')
            
            # add ZT time axis
            ax4 = ax3.twiny()
            ax4.set_xticks(trange)
            ax4.set_xticklabels(toZT(trange).astype(int))
            ax4.set_xlim(ax3.get_xlim())
            ax4.xaxis.set_ticks_position('bottom'); ax4.xaxis.set_label_position('bottom')
            ax4.spines['bottom'].set_position(('outward', 40))
            ax4.set_xlabel('ZT time')
            
            summit_pngs.append(plot_name)
            plt.savefig(plot_name, bbox_inches='tight', dpi=100, pad_inches=0.02)
            plt.close()
            
            
        if do_non_summiter_calling:
            # check there are some left!
            if len(eligible_ROIs) < n_extra_non_summiters:
                printf('\t\t\t!No more eligible ROIs as non-summiters', run_log) 
                break
        
            # get 3 times as many candidate non-summiters as n_extra_non_summiters        
            candidate_non_summiters_num = n_extra_non_summiters*3
            candidate_non_summiter_is = [x for x in argsorted_non_summiter_scores if x in eligible_ROIs][:candidate_non_summiters_num]
            # now pick n_extra_nonsummiters randomly, so that the same candidates
            # don't keep passing along every time, but retain the top candidate
            pred_non_summiter_is = np.random.choice(candidate_non_summiter_is[1:],
                                                    n_extra_non_summiters,
                                                    replace=False)
            pred_non_summiter_is[0] = candidate_non_summiter_is[0]
                       
            # plot all of the candidates
            for pred_non_summiter_i in pred_non_summiter_is:
                pred_score = non_summiter_scores[pred_non_summiter_i]
                
                # get y, spd for ROI
                ypos = ed['ypos'][:grab_index, pred_non_summiter_i] 
                spd = ed['speed'][:grab_index, pred_non_summiter_i]
                
                # plot non-summiter
                plt.figure(figsize=(12,8))
                plt.suptitle('CPU: {}, ROI {} ({}), predicted non-summit score: {:.2f} at {:02d}:{:02d}:{:02d}'.format(
                        cpu_name, 
                        pred_non_summiter_i+1, 
                        ROI_pos_on_board(pred_non_summiter_i+1),
                        pred_score, 
                        *hr_min_sec_from_hr_float(cur_time)))
                ax1, ax2, ax3 = plot_classifier_traj(t=ts[:grab_index], 
                     ypos=ypos, spd=spd,
                     pred_times=classify_times[:classify_i], 
                     pred_probs=predYprobs[:classify_i, pred_non_summiter_i, :],
                     dic_class_to_i=dic_class_to_i, dic_i_to_class=dic_i_to_class)
                plot_name = os.path.join(non_summit_plot_dir, 
                                         'time{}_{}_{}_score{:.0f}_roi{}.png'.format(
                                                 *hr_min_sec_from_hr_float(cur_time),
                                                 round(pred_score*1000),
                                                 pred_non_summiter_i+1))
                trange = np.arange(np.round(ts[0]), ts[grab_index]+0.001, 1.0)
                ax3.set_xticks(trange)
                ax3.set_xlabel('local time')
                
                # add ZT time axis
                ax4 = ax3.twiny()
                ax4.set_xticks(trange)
                ax4.set_xticklabels(toZT(trange).astype(int))
                ax4.set_xlim(ax3.get_xlim())
                ax4.xaxis.set_ticks_position('bottom'); ax4.xaxis.set_label_position('bottom')
                ax4.spines['bottom'].set_position(('outward', 40))
                ax4.set_xlabel('ZT time')
                plt.savefig(plot_name, bbox_inches='tight', dpi=100)
                non_summit_pngs.append(plot_name)
                plt.close()
                
                # save to table
                out_line += [pred_non_summiter_i+1, ROI_pos_on_board(pred_non_summiter_i+1)]
            
            # get top non_summiter
            top_non_summiter_i = pred_non_summiter_is[0]
            # get its score, add it to count
            top_pred_score = non_summiter_scores[top_non_summiter_i]
            current_called_non_summiter_ROI_is.append(top_non_summiter_i)
            n_total_called_non_summiters += 1
            printf('\t\t\t!predicted non-summiter: ROI {}, score: {:.3f}'.format(top_non_summiter_i+1, top_pred_score), run_log) 
            # remove this top non-summiter from subsequent steps
            eligible_ROIs = eligible_ROIs[eligible_ROIs != top_non_summiter_i]
        
        # add line to out_table
        out_table.append(out_line)
        # save csv
        out_df = pd.DataFrame(out_table, columns=out_table_cols)
        out_df.to_csv(out_table_path, index=False)
    
    if do_non_summiter_calling:
        all_called_non_summiter_ROI_is = np.concatenate((all_called_non_summiter_ROI_is, 
                     current_called_non_summiter_ROI_is)).astype(int)
            
    t_plot = time.time()
    printf('\tElapsed {:.3f} sec: summiting ROIs plotted'.format(t_plot - t0), run_log)
    
    # optionally send email
    if n_current_called_summiters > 0:
        printf('updating at {:02d}:{:02d}:{:02d} with {}, total is {}'.format(
                *hr_min_sec_from_hr_float(cur_time), 
                n_current_called_summiters, 
                n_total_called_summiters), pred_log, False)
        
        # send update email
        if do_email:
            
            # write content
            content = 'We have {} summiting at {:02d}:{:02d}:{:02d} on {}\n\n'.format(
                        n_current_called_summiters, 
                        *hr_min_sec_from_hr_float(cur_time),
                        cpu_name)
            content += 'Summiting ROIs: {}\n'.format([i+1 for i in current_called_summiter_ROI_is])
            if do_non_summiter_calling:
                content += 'Non-summiting ROIs: {}\n'.format([i+1 for i in current_called_non_summiter_ROI_is])
            content += '\nPreviously called summiting ROIs: {}\n'.format([i+1 for i in all_called_summiter_ROI_is[:-n_current_called_summiters]])
            if do_non_summiter_calling:
                content += 'Previously called non-summiting ROIs: {}\n'.format([i+1 for i in all_called_non_summiter_ROI_is[:-n_current_called_summiters]])
            content += '\nWe are up to {} summiters now.\n'.format(n_total_called_summiters)+\
                        'CPU: {}\n'.format(cpu_name)+\
                        'Experiment: {}'.format(centroid_bin_file)
                        
            # send separate email for each attachment
            for ei in range(n_current_called_summiters):
                
                attachment_files = [summit_pngs[ei]]
                if do_non_summiter_calling:
                    nonsum_img_lb = ei*n_extra_non_summiters
                    nonsum_img_ub = nonsum_img_lb + n_extra_non_summiters
                    attachment_files += non_summit_pngs[nonsum_img_lb:nonsum_img_ub]
                attachment_files.append(out_table_path)
                
                email_sent, email_status = send_email(email_recip, 
                       subject='{}: summiting, classifier run: {}'.format(cpu_name, run_time_str),
                       content=content,
                       filesToSend=attachment_files)
                email_statuses.append((cur_time, email_status))
                
        t_email = time.time() 
        no_email_word = 'not ' if (not do_email or email_sent == 0) else ''
        printf('\tElapsed {:.3f} sec: {}emailed'.format(t_email - t0, no_email_word), run_log)          
        
        
    # time it 
    end_time = time.time()
    printf('\tElapsed {:.3f} sec: done with frame\n\tROIs processed.\n'.format(end_time-t0), run_log)