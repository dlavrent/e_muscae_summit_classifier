# -*- coding: utf-8 -*-
"""
Various time-related utilities
"""

import datetime
import re

def get_cur_day():
    '''
    Returns day of the month
    Useful for determining when to add 24 hours 
    (i.e. when experiment crosses past midnight)
    '''
    return datetime.datetime.now().day

def get_day_thru_sec():
    '''
    Returns list of form [year, month, day, hour, minute, second]
    '''
    return list(datetime.datetime.now().timetuple())[:6]

def hr_min_from_hr_float(hr_float):
    '''
    Given an hour float, returns nearest hour:minute
    (e.g. 17.75 is 17:45, returns [17,45])
    '''
    cur_hr, cur_min = [int(p) for p in divmod(hr_float*60, 60)]
    return cur_hr, cur_min

def hr_min_sec_from_hr_float(hr_float):
    '''
    Given an hour float, returns nearest hour:minute:second
    (e.g. 17.75 is 17:45:00, returns [17,45,0])
    '''
    cur_hrf, cur_minf = [p for p in divmod(hr_float*60, 60)]
    cur_hr = int(cur_hrf)
    cur_min, cur_sec = [int(p) for p in divmod(cur_minf*60., 60)]
    return [cur_hr, cur_min, cur_sec]

def frame_to_time(frame):
    return frame/60/60/3

def time_to_frame(time):
    return time*60*60*3

def local_time_from_frame(frame, start_time):
	'''
	Given experiment start time in hours (start_time),
	returns local time of frame of interest in hours.
	HARDCODED FRAME RATE (3) AND UNIT (HERTZ)
	''' 
	return start_time + frame_to_time(frame)

def frame_from_local_time(local_time, start_time):
	'''
	Given experiment start time in hours (start_time),
	determines number of frames since start of 
	experiment for a frame of interest.
	HARDCODED FRAME RATE (3) AND UNIT (HERTZ)
	'''
	return time_to_frame(local_time - start_time)

def get_cur_time(add_offset=0):
    '''
    Gets time reading from datetime (in hours)
    and adds optional offset (in hours)
    e.g. at 14:30 it'd return 14.5
    '''
    t = datetime.datetime.now()
    return t.hour + t.minute/60 + t.second/60/60 + add_offset

def get_exp_time_from_filename(file):
    '''
    Extracts time from filename typical of margo outputs
    '''
    exp_date, exp_time = re.findall('([0-9]{2}-[0-9]{2}-[0-9]{4})-([0-9]{2}-[0-9]{2}-[0-9]{2})', file)[0]
    exp_hr, exp_min, exp_sec = [int(x) for x in exp_time.split('-')]
    exp_time = exp_hr + exp_min/60 + exp_sec/3600
    return exp_time

