'''
Usefull data processing utility functions
'''
import numpy as np
from utils.time_utils import frame_from_local_time, frame_to_time
from sklearn.metrics import  auc, roc_curve
import bottleneck as bn

#############################
### Classifier output helpers
#############################

def printf(msg, outfile, print_console=True):
    '''
    Prints msg to file, and/or to console
    '''
    with open(outfile, 'a') as f: 
        print(msg, file=f)
    if print_console:
        print(msg)

def ROI_pos_on_board(ROI):
    '''
    Converts an ROI (1-128) to 
    (row, position) on the board
    '''
    roi_row = ROI // 32 + 1 - 1*(ROI % 32 == 0)
    roi_well = ROI % 32  + 32*(ROI % 32 == 0)
    return '({})-{}'.format(roi_row, roi_well)


#############################
# Classifier training helpers
#############################

def get_frames(base_frame, params):
    '''
    Given a frame of interest (base_frame), returns two sets/
    windows of frames: prev_frames and curr_frames.
    Settings for window sizes and how to space the sets
    (evenly or logarithmically spaced) are in params.
    '''

    # current time window: [base_frame-width, base_frame]
    # evenly spaced, with NFULL frames
    curr_frames = np.linspace(
        max(0, base_frame-params['CURR_WINDOW_WIDTH']),
        base_frame,
        params['NCURR']).astype(int)

    # previous time window

    if not params['LOG_SPACING']:
        # take a window from [0, base_frame - width],
        # evenly spaced, with NPREV frames
        prev_frames = np.linspace(
            params['PREV_LB'],
            max(params['PREV_LB'], base_frame-params['CURR_WINDOW_WIDTH']),
            params['NPREV']+1)[:-1].astype(int) # do a [1:] indexing to make sure 
                                                # you don't capture the start of 
                                                # curr_frames (base_frame - width)
    else:
        # take a window from [PREV_LB, base_frame-width],
        # logarithmically spaced, with NPREV frames
        prev_frames = np.logspace(
            np.log(params['PREV_LB']), 
            max(np.log(params['PREV_LB']), np.log(base_frame-params['CURR_WINDOW_WIDTH'])), 
            params['NPREV']+1, base=np.e)[1:].astype(int) # again do [1:] indexing to make sure 
                                                          # you don't capture the start of 
                                                          # curr_frames (base_frame - width)
        # make sequence more dense near the curr_frames
        prev_frames = prev_frames[-1] + params['PREV_LB'] - prev_frames[::-1]

    return prev_frames, curr_frames

def get_tau_frame_for_roi(roi, t_df):
    '''
    Returns the best tau for an roi
    '''
    # put in checks here for non NaN?
    return t_df['Best_tau'][roi-1]


def get_last_mvmt_for_roi(roi, t_df):
    '''
    Returns the best tau for an roi
    '''
    # put in checks here for non NaN?
    return t_df['Last_mvmt'][roi-1]

def make_X_from_frames_roi(base_frame, grab_frames, ypos, spd, t_ed):
    '''
    For an ROI and frame of interest (base_frame), and frames 
    corresponding to associated prev and curr frames (grab_frames),
    takes relevant time, ypos, and speed data to make processed data.
    '''
    return np.concatenate(([t_ed['ts'][base_frame]], 
                        ypos[grab_frames],
                        spd[grab_frames]
                        ))

def make_continuous_y_from_frames_roi(curr_frames, roi, t_ed, t_df, params):
    '''
    For an ROI and current frames (curr_frames), if end-state is 
    cadaver, returns stage (pre-/dur-/post-summiting), or returns
    alive if end-state is alive.
    '''
    state = t_df['Res'][roi-1] # 'Alive', 'Cadaver', or 'NI'
    decision_frame = curr_frames[-1]
    tau_frame = get_tau_frame_for_roi(roi, t_df)
    last_mvt_frame = get_last_mvmt_for_roi(roi, t_df)
    if state == 'Alive':
        return params['alive_val']
    elif state == 'Cadaver':
        if np.isnan(tau_frame):
            return np.nan
        return frame_to_time(decision_frame - tau_frame)
    else:
        return np.nan

def label_cad_y(t, lb, ub):
    '''
    Given a time for consideration t, and
    lower/upper bounds (lb, ub) for summiting,
    outputs whether t is pre-, during-, or post-summit
    '''
    if t < lb:
        return 'pre'
    if lb <= t and t <= ub:
        return 'dur'
    else:
        return 'post'

def make_y_from_frames_roi(curr_frames, roi, t_ed, t_df, params):
    '''
    For an ROI and current frames (curr_frames), if end-state is 
    cadaver, returns stage (pre-/dur-/post-summiting), or returns
    alive if end-state is alive.
    '''
    state = t_df['Res'][roi-1] # 'Alive', 'Cadaver', or 'NI'
    decision_frame = curr_frames[-1]#params['NCURR']*3//4-1]
    tau_frame = get_tau_frame_for_roi(roi, t_df)
    last_mvt_frame = get_last_mvmt_for_roi(roi, t_df)
    if state == 'Cadaver':
        if np.isnan(tau_frame):
            return 'NI'
        return label_cad_y(decision_frame, 
                tau_frame, last_mvt_frame)
    return state # if not Cadaver, then Alive


def make_X_y_from_frame(base_frame, roi, ypos, spd, t_ed, t_df, params):
    '''
    For one ROI, given a frame of interest (base_frame), 
    extracts frames for previous and current windows, 
    then makes processed X, y matrices for the ROI.
    '''
    # get relevant frames
    prev_frames, curr_frames = get_frames(base_frame, params)
    grab_frames = np.concatenate((prev_frames, curr_frames))

    # make X, y
    X_ex = make_X_from_frames_roi(base_frame, grab_frames, ypos, spd, t_ed)
    if params['do_continuous']:
        y_ex = make_continuous_y_from_frames_roi(curr_frames, roi, t_ed, t_df, params)
    else:
        y_ex = make_y_from_frames_roi(curr_frames, roi, t_ed, t_df, params)

    
    return X_ex, y_ex

def make_X_y_from_expmt(t_ed, t_df, params):
    '''
    Goes through each ROI of experiment (t_ed, t_df)
    and constructs processed X, y data using the helper
    functions above.
    '''
    expmt_start_time = t_df['start_time'][0]
    X = []; y = []
    log = []
    for roi in t_df.ROI.astype(int):
        state = t_df['Res'][roi-1]
        tau_frame = t_df['Best_tau'][roi-1]
        ypos = t_ed['ypos'][:, roi-1]
        spd = t_ed['speed'][:, roi-1]
        if params['do_mov_avg']:
            ypos = bn.move_mean(ypos, window=params['wl'], min_count=1)
            spd = bn.move_mean(spd, window=params['wl'], min_count=1)
        if state != 'NI': # skip NI's
            # also skip cadavers with no tau
            if state == 'Cadaver' and np.isnan(tau_frame): 
                pass
            else: # otherwise, we're good
                # draw EX_PER_TRAJ frames uniformly
                frames_to_add = np.random.uniform(         
                    frame_from_local_time(
                        params['MIN_LOCAL_TIME'], expmt_start_time),         
                    len(t_ed['ts']), 
                    params['EX_PER_TRAJ']).astype(int)
                # for each frame, make an X, y instance
                for f in frames_to_add:
                    X_ex, y_ex = make_X_y_from_frame(f, roi, ypos, spd,
                                                     t_ed, t_df,
                                                     params)
                    # ignore NaN filled vectors
                    if ~np.isnan(X_ex[1]):
                        X.append(X_ex)
                        y.append(y_ex)
                        log.append((roi, f))
    X = np.vstack(X); y = np.array(y)
    return X, y, np.array(log)

def y_to_one_hot(y, label_is):
    '''
    Convert categorical y to one hot encoding
    according to the order in label_is
    '''
    oneh = np.zeros((len(y), len(label_is)))
    for i in range(len(y)):
        oneh[i, label_is[y[i]]] = 1
    return oneh

def get_fprs_tprs_aucs(clf, VAL_X, VAL_Y, labs, class_i_labels_d):
    '''
    Utility function for making ROC plot
    '''
    y_val_predprobs = clf.predict_proba(VAL_X)
    y_val_1h = y_to_one_hot(VAL_Y, class_i_labels_d)
    fprs = []; tprs = []; aucs = []
    for lab in labs:
        labi = class_i_labels_d[lab]
        fpr, tpr, _ = roc_curve(y_val_1h[:, labi], y_val_predprobs[:, labi])
        aucs.append(auc(fpr, tpr))
        fprs.append(fpr); tprs.append(tpr)
    return fprs, tprs, aucs


def count_pred_true_matrix(df, predstyle):
    '''
    Function for building confusion matrix in training/test_classification.ipynb
    Given a dataframe, df, with columns corresponding to classification rules,
    and rows corresponding to individual flies, 
    builds alive/during-summiting confusion matrix
    '''
    true_alive_pred_alive = df[(df.Res == 'Alive') & (np.isnan(df[predstyle]))].shape[0]
    true_cad_pred_alive = df[(~np.isnan(df.tau_hr)) & (np.isnan(df[predstyle]))].shape[0]
    true_alive_pred_dur = df[(df.Res == 'Alive') & (~np.isnan(df[predstyle]))].shape[0]
    true_cad_pred_dur = df[(~np.isnan(df.tau_hr)) & (~np.isnan(df[predstyle]))].shape[0]
    return np.array([[true_alive_pred_alive, true_cad_pred_alive],
              [true_alive_pred_dur, true_cad_pred_dur]])

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