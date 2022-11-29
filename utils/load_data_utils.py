# -*- coding: utf-8 -*-
"""
Utilities for loading/processing data from 
    binary files (outputted from margo), or
    _margoConvert.mat files
"""

import array
import numpy as np
import pandas as pd
import hdf5storage
import re
import os
from utils.time_utils import get_exp_time_from_filename


##################################################
# Common to both binary/_margoConvert file reading
##################################################  


MAX_N_FRAMES = 3*60*60*48 # 48 hours


def forward_fill_nans(arr):
    '''
    Utility to fill NaN's in an array with the previous
    non-NaN value, and also fill in leading NaN's with
    the first non-NaN. Modeled from
        https://stackoverflow.com/a/41191127
    '''
    # forward fill NaNs with previous non-NaN
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    arr = arr[idx]
    # now fill in the beginning of NaN's
    first_nonnan = np.argmin(np.isnan(arr))
    arr[:first_nonnan] = arr[first_nonnan]
    return arr


##################################################
############ For loading in data from binary files
##################################################  


def read_in_centroids_and_time(centroid_bin_file, time_bin_file, NROI=128):
    '''
    Given binary files for centroid and time data,
    loads in the array for centroids (in shape n_frames * 2 * nROI),
    loads in the array for time (in shape n_frames*1)
    '''
    # read in time binary file 
    try:
        time_array = array.array("f")
        time_array.fromfile(open(time_bin_file, mode='rb'), MAX_N_FRAMES)
        # this is a trick! Let n be large, like the max number of frames. Then,
        # array.fromfile will read in up to n items. if n exceeds the number
        # of items in the actual file, a warning gets raised, but it'll still
        # load in the real number of items... so to suppress the warning, I
        # put this in a try: except and just return the object loaded with
        # however true items there are inside.
    except EOFError:
        time_array = np.array(time_array)
        
    if len(time_array) == 0:
        raise ValueError('time binary file is empty!')
           
    # read in centroids binary file with same process
    try:
        centroids = array.array("f")
        centroids.fromfile(open(centroid_bin_file, mode='rb'), NROI*2*MAX_N_FRAMES)
        
    except EOFError:
        n_frames = int(len(centroids) / NROI / 2)
        centroids = np.array(centroids).reshape((n_frames, 2, NROI))
        
    if len(centroids) == 0:
        raise ValueError('centroids binary file is empty!')

    return centroids, time_array

def get_spd_from_centroids_bin(centroids):
    '''
    Given centroids data (n_frames * 2 * nROI),
    returns speed (n_frames * nROI) by computing 
    COM distance between two consecutive frames.
    '''
    # initialize with zeros
    spds = np.zeros(shape=(centroids.shape[0], centroids.shape[2]))
    # spd(i) = sqrt( [ (x(i) -x(i-1) ]^2 + [ (y(i) -y(i-1) ]^2 )
    spds[1:, :] = np.sum(np.diff(centroids, axis=0)**2, axis=1)**(1/2)
    # finally, fill NaN's with 0
    return np.nan_to_num(spds) 
    

def get_ypos_and_frame_times_bin(centroids, time_array, expmt_start_time):
    '''
    Given centroids and time_array 
    (the time offset between frames outputted by margo tracking), 
    returns ypos and true times of frames (in hours)
    '''
    ypos = centroids[:, 1, :]
    frame_times = expmt_start_time + np.cumsum(time_array) / 60 / 60
    
    return ypos, frame_times


def clean_ypos_from_raw_bin(ypos_raw):
    '''
    Given the y positions from margo, which is full of NaN's,
        forward fill the NaN's, 
        scale to 0-1, 
        and flip so that bottom is food
    '''
    # this takes ~1 second
    ypos = np.apply_along_axis(forward_fill_nans, 0, ypos_raw)
    # make sure there's at least one value
    # this takes ~ second
    #more_than_one_val = np.array(
    #    [len(np.unique(ypos[:, x])) > 1 for x in range(ypos.shape[1])])
    #ypos[:, ~more_than_one_val] = np.nan
    # scale each ROI ypos to 0 and 1 using max/min of trajectory
    where_not_nan = ~np.all(np.isnan(ypos), axis=0)
    roi_maxs = np.max(ypos[:, where_not_nan], axis=0)
    roi_mins = np.min(ypos[:, where_not_nan], axis=0)
    # scale to 0 and 1 and flip so that food on bottom
    ypos[:, where_not_nan] = 1 - \
        (ypos[:, where_not_nan] - roi_mins) / (roi_maxs - roi_mins)
    return ypos



def make_ed_from_bin_files(centroid_bin_file, time_bin_file, NROI=128):
    '''
    Puts everything together, reading in centroid, time binary files,
    and returning a dictionary with the relevant information
    (speed array, ypos array, times of frames)
    '''
    # read in the binary data
    centroids, time_array = read_in_centroids_and_time(centroid_bin_file, time_bin_file, NROI)
    # parse the experiment start time from the filenames
    expmt_start_time = get_exp_time_from_filename(centroid_bin_file)
    # use centroids to get raw ypos, and 
    # use time array to get frame times
    ypos_raw, frame_times = get_ypos_and_frame_times_bin(centroids, time_array, expmt_start_time)
        
    # clean ypos, compute speed from centroids
    ypos = clean_ypos_from_raw_bin(ypos_raw)
    spd = get_spd_from_centroids_bin(centroids)
    
    # return relevant info in dictionary
    return {'speed': spd,
          'ypos_raw': ypos_raw,
          'ypos': ypos,
          'start_time': expmt_start_time,
          'ts': frame_times,
          'NROI': ypos.shape[1]}


##################################################
##### For loading in data from _margoConvert files
##################################################    

def d_from_expmt_mat(f):
    '''
    Opens a _margoConvert.MAT file f and converts to
    a Python dictionary. Has the same exact fields/
    structure as the 'exmpt' MATLAB struct.
    Note that this conversion just uses hardcoded
    numbering for indexing, so if the order of fields
    changes in the MATLAB struct, update it here too!
    '''
    # load _margoConvert.MAT file
    mat = hdf5storage.loadmat(f)
    # populate dictionary
    expmt_d = {}
    expmt_d['Centroid'] = {'data': mat['expmt'][0][0][0][0]}
    expmt_d['Speed'] = {'data': mat['expmt'][0][1][0][0]}
    expmt_d['parameters'] = {'target_rate': mat['expmt'][0][2][0][0][0][0],
                        'mm_per_pix': mat['expmt'][0][2][0][1][0][0]}
    expmt_d['fLabel'] = mat['expmt'][0][3][0][0]
    expmt_d['ROI'] = {'bounds': mat['expmt'][0][4][0][0]}
    expmt_d['nTracks'] = mat['expmt'][0][5][0][0]
    return expmt_d


def expmt_dict_from_dmat(dmat, dofill=True):
    '''
    Converts dmat (Python dic representation of matlab struct)
    to a more useful version with scaled ypos data, experiment 
    date/time, and time series
    '''
    ed = {}
    ed['speed'] = dmat['Speed']['data']
    # keep raw ypos data
    ed['ypos_raw'] = dmat['Centroid']['data'][:, 1, :]
    # fill in the NaNs
    ed['ypos'] = np.apply_along_axis(forward_fill_nans, 0, ed['ypos_raw'])
    # make sure there's at least one value
    more_than_one_val = np.array(
        [len(np.unique(ed['ypos'][:, x])) > 1 for x in range(ed['ypos'].shape[1])])
    ed['ypos'][:, ~more_than_one_val] = np.nan
    # scale each ROI ypos to 0 and 1 using max/min of trajectory
    where_not_nan = ~np.all(np.isnan(ed['ypos']), axis=0)
    roi_maxs = np.max(ed['ypos'][:, where_not_nan], axis=0)
    roi_mins = np.min(ed['ypos'][:, where_not_nan], axis=0)
    # scale to 0 and 1 and flip so that food on bottom
    ed['ypos'][:, where_not_nan] = 1 - \
        (ed['ypos'][:, where_not_nan] - roi_mins) / (roi_maxs - roi_mins)
    # find experiment date, time
    exp_date, exp_time = re.findall( \
            '^([0-9]{2}-[0-9]{2}-[0-9]{4})-([0-9]{2}-[0-9]{2}-[0-9]{2})', \
            dmat['fLabel'])[0]
    exp_hr, exp_min, exp_sec = [int(x) for x in exp_time.split('-')]
    ed['start_time'] = exp_hr + exp_min/60 + exp_sec/3600    
    # make array of time points (use the actual times?)
    expmt_len = ed['speed'].shape[0]
    ed['ts'] = ed['start_time'] + np.arange(expmt_len)/60/60/3
    return ed


def get_ed_from_exp_dir(exp_dir):
    '''
    Loads in margoConvert file and returns an
    experiment dictionary ("ed") with ypos,
    speed, time array.
    '''
    # find margoConvert file
    exp_files = os.listdir(exp_dir)
    try:
        margo_file = [x for x in exp_files if '_margoConvert' in x][0]
    except:
        print('No margoConvert file here!')
        raise
    margo_file = os.path.join(exp_dir, margo_file)
    # load in experiment info
    dmat = d_from_expmt_mat(margo_file)
    ed = expmt_dict_from_dmat(dmat)
    return ed


def load_surv_df(surv_data_file, exp_time):
    '''
    Utility for get_df_from_exp_dir function.
    Loads the survival_data excel into a dataframe,
    notably adding a final state column ('Res'), adding
    time tau in hours, and adding experiment start time
    '''

    # load in survival_data excel file
    surv_df = pd.read_excel(surv_data_file)

    # delete blank columns
    surv_df = surv_df.drop(columns=[x for x in surv_df.columns if x[:7] == 'Unnamed'])
    
    # code 'Res' to outcome
    surv_df['Res'] = ''
    surv_df.loc[(surv_df.Status == 1) & (surv_df.Outcome == 0), 'Res'] = 'Alive'
    surv_df.loc[(surv_df.Status == 0) & (surv_df.Outcome == 0), 'Res'] = 'NI'
    surv_df.loc[(surv_df.Status == 0) & (surv_df.Outcome == 1), 'Res'] = 'Cadaver'

    # convert best tau from frames since start of experiment
    # to local time (in hours)
    surv_df['tau_hr'] = np.nan 
    surv_df.loc[surv_df.Best_tau > 0, 'tau_hr'] = \
        exp_time + surv_df.Best_tau[surv_df.Best_tau > 0] / 3600 / 3

    # add time of experiment
    surv_df['start_time'] = exp_time
    
    # make sure keep only rows with an ROI label
    surv_df = surv_df[np.isfinite(surv_df['ROI'])]
    return surv_df


def get_df_from_exp_dir(exp_dir):
    '''
    Given experiment directory, loads/processes 
    the survival data file, and also uses the 
    margoConvert file to get experiment start time
    '''
    
    # find margoConvert file
    exp_files = os.listdir(exp_dir)
    try:
        margo_file = [x for x in exp_files if '_margoConvert' in x][0]
    except:
        print('No margoConvert file here!')
        raise
    # use the margoConvert filename to extract experiment start time
    margo_file = os.path.join(exp_dir, margo_file)
    exp_time = get_exp_time_from_filename(margo_file)

    # find survival data file
    try:
        surv_data_file = [x for x in exp_files if 'survival_data' in x][0]
    except:
        print('No survival_data file here!')
        raise
    # load survival data into dataframe
    surv_data_file = os.path.join(exp_dir, surv_data_file)
    return load_surv_df(surv_data_file, exp_time)


def get_data_from_exp_dir(exp_dir):
    '''
    Given an experiment directory with margoConvert 
    and survival_data files, returns an experiment
    dictionary (ed) and survival dataframe (surv_df), 
    used for training
    '''

    ed = get_ed_from_exp_dir(exp_dir)
    surv_df = get_df_from_exp_dir(exp_dir)
    
    return ed, surv_df