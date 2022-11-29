# -*- coding: utf-8 -*-
"""
This file:
- sets the location of the data to read in (either in real time or post-factum)
- sets settings for classification (what trained model to use, what frequency to classify with, etc.)
- sets email recipients, if desired
- is read in by run_classifier.py to perform classification
"""

import os
import sys
file_path = os.path.abspath(__file__)
project_dir = os.path.abspath(os.path.join(file_path, os.pardir, os.pardir))
print('project directory: {}'.format(project_dir))

input_d = {
    # your run name, will be used in the output names
    'run_name': 'test_classifier_run',
    
    # computer name
    'cpu_name': 'test_computer',
    
    # identify home directory (and it's where outputs will be saved)
    'home_dir': project_dir,

    # classifier files
    # if experiment is being collected in real-time,
    # note that the centroid binary file updates continuously,
    # while the time binary file updates about every 6 minutes
    'clf_name': 'clf_used_in_experiments.p',
    'params_name': 'params_used_in_experiments.p',

    # data to read, need to specify centroid and time MARGO binary files
    'centroid_bin_file': r'../Data/All raw behavior data/11-05-2019-18-38-20__Circadian_CsWF-BoardC10_MF_Emuscae_1-128_Day3/raw_data/11-05-2019-18-38-20__centroid.bin',
    'time_bin_file': r'../Data/All raw behavior data/11-05-2019-18-38-20__Circadian_CsWF-BoardC10_MF_Emuscae_1-128_Day3/raw_data/11-05-2019-18-38-20__time.bin',
    
    # email settings
    # can be a string or list
    'do_email': 0,
    'email_recip': [],

    # do in real time (pause for new data to come in), 
    # or use already-collected data and skip along classification frames?
    'real_time': 0,
    
    # identify non-summiters in addition to summiters?
    'do_non_summiter_calling': 1,
    'n_extra_non_summiters': 5, # for every called summiter, how many top non-summiting score ROIs to plot?

    # save plots? (can take a long time if many summiters called at once)
    'do_plot': True, 
    
    # do adjustment for daylight savings time?
    # if fall-winter, set this to 1 (no DST), if spring-summer, set this to 0 (DST)
    # most of the data is trained from summer!
    # when we "fall back" and become an hour early, what appears as hour 33 to us 
    # is really 34 for the flies, so adjust the ZT time accordingly
    'NO_DST': 1, 
    
    # time settings
    'TIME_OFFSET0': 12, # (hours)
                        # useful for code development, shifts current time by TIME_OFFSET0 hours 
                        # for instance, if you have existing data with summiting at, say, 34:00,
                        # but the current local time is 22:00, set this to 12 to simulate how 
                        # the classifier would work in real time
    'SLEEP_TIME': 5,    # (seconds)
                        # how often to check what time it is and decide whether to classify

    # classification time settings
    'classify_freq': 4/60, # (hours)
                           # how often to perform classification 
                           # (process ROIs into features for random forest, classify, plot, email)
    'classify_start': 29,  # (hours)
                           # when to start classification (starting at 8:30pm of day 1 => 20.5)
    'classify_stop': 36,   # (hours)
                           # when to stop classification (2pm of day 2 => 24+14=38)

    # summit call setting
    'Yprob_consec_frames': 3, # how many consecutive classification frames 
                              # (with frequency classify_freq)
                              # does a condition need to hold 
                              # for an ROI to be called summiting? 
                              # the condition is defined in run_classifier.py
}