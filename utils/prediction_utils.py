# -*- coding: utf-8 -*-
"""
Utilities for converting predicted class probabilities
    into more actionable decisions -- i.e. for locating
    first appearance of some event (e.g. first time
    P(dur) > P(alive)), or for when there's a stretch 
    of some condition (P(dur) > P(alive) for 3 straight minutes)
"""

import numpy as np

#############################
######### Make training data
#############################


def roi_first_app_of_condition(margo_frames, condition):
    '''
    Given array of frames (margo_frames) and 
    some true/false condition of size (nFrames * nROI), 
    returns first appearance of condition for each ROI
    (or NaN otherwise)
    '''
    # get the frame and ROI positions where condition is true
    timeis, roiis = np.where(condition)
    # return condition's first appearance per ROI
    roiis, roii_firstapps = np.unique(roiis, return_index=True)
    timeis = timeis[roii_firstapps]

    # return full array of size nROI, with first appearance
    # if condition is present, or else NaN
    roi_durs = np.full([condition.shape[1],], np.nan)
    for i in range(len(roiis)):
        roii = roiis[i]
        timei = timeis[i]
        roi_durs[roii] = margo_frames[timei]
        
    return roi_durs

def one_runs(a):
    '''
    Utility for longest_1_stretch_end,
    Given an array with true (1) / false (0) values,
    outputs the ranges of consecutive trues
    i.e. if a = [0,1,0,0,1,1,1,0,0,0,1,1],
         returns ([1,2], [4,7], [10,12])
    Reference: https://stackoverflow.com/questions/24885092/finding-the-consecutive-zeros-in-a-numpy-array
    '''
    # Create an array that is 1 where a is 1, and pad each end with an extra 0.
    is_one = np.concatenate(([0], np.equal(a, 1), [0]))
    absdiff = np.abs(np.diff(is_one))
    # Returns start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def longest_1_stretch_end(a, thresh=10):
    '''
    Given ranges of consecutive trues in array a,
    returns a new array new_a, in which only one
    span of consecutive trues is present -- the
    first span that has length >= thresh
    (if no such span exists, returns all zeroes)
    '''
    ranges = one_runs(a)
    new_a = np.zeros(len(a))
    if len(ranges) == 0:
        return new_a
    where_longer_than_thresh = np.where(ranges[:, 1] - ranges[:, 0] > thresh)[0]
    if len(where_longer_than_thresh) > 0:
        lb, ub = ranges[where_longer_than_thresh[0]]
        new_a[lb+thresh] = 1
    return new_a