import mne
import os

import pandas as pd
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import Covariances

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from numpy.random import seed
seed(2002012)


########################################
# DEFINE FUNCTIONS FROM INRIA PIPELINE #
########################################

def preprocess(raw, steps={}):
    """ preprocess the data"""
    assert isinstance(steps, dict), "steps must be a dictionary"
    raw.load_data()
    if "drop_channels" in steps.keys():
        # remove the wanted channels
        for channel in steps["drop_channels"]:  # for each channel to be dropped...
            if channel in raw.ch_names:  # ensure that it is actually a channel in the data
                raw.drop_channels(channel)

    if "filter" in steps.keys():
        assert isinstance(steps["filter"], list), "filter parameters must be a list in the form [l_freq,h_freq]"
        raw.filter(steps["filter"][0], steps["filter"][1])

    return raw


def epoching(dict, key_session=[], steps_preprocess=None, key_events={"769": 0, "770": 1}):
    """From the dictionary of mne.rawGDF extract all the epochs selected with Key_session
     Return the epochs list as X and the label as Y"""

    #---------------------------------------------
    tmin = steps_preprocess["tmin"]
    tmax = steps_preprocess["tmax"]
    length_epoch = steps_preprocess["length"]
    overlap = steps_preprocess["overlap"]
    #---------------------------------------------
    X = None
    Y = None

    for key in key_session:
        if steps_preprocess is not None:
            _ = preprocess(dict[key], steps_preprocess)

        epoch = mne.Epochs(dict[key], mne.events_from_annotations(dict[key], key_events)[0], tmin=-1, tmax=5,
                           baseline=(None, 0),  verbose="CRITICAL")

        list_start = np.arange(tmin, (tmax + overlap) - length_epoch, overlap)
        list_stop = np.arange(tmin+length_epoch, (tmax+overlap), overlap)

        for start, stop in zip(list_start, list_stop):
            if X is None:
                X = epoch.get_data(tmin=start, tmax=stop)
                Y = epoch.events[:, 2]

            else:
                X = np.append(X, epoch.get_data(tmin=start, tmax=stop), axis=0)
                Y = np.append(Y, epoch.events[:, 2], axis=0)

    return X, Y


def test_pipeline_within_session(pipelines, session, steps_preprocess=None):
    """ Take in input the different pipelines to test and return the corresponding classification accuracy"""
    accuracy = pd.DataFrame(np.zeros((len(session), len(pipelines))), index=session, columns=pipelines.keys())

    for subject in session:
        train_key = {k: v for k, v in dic_data.items() if subject not in k}  # train on all subject's data EXCEPT one
        test_key = {k: v for k, v in dic_data.items() if subject in k}  # test on that subject

        print(subject)
        X_train, Y_train = epoching(dic_data, train_key, steps_preprocess)  # X = epochs, Y = labels
        X_test, Y_test = epoching(dic_data, test_key, steps_preprocess)  # same as above, but test set

        for classifier in pipelines.keys():
                pipelines[classifier].fit(X_train, Y_train)

                if steps_preprocess["score"] == "TAcc":

                    #---------------------------------------------
                    tmin = steps_preprocess["tmin"]
                    tmax = steps_preprocess["tmax"]
                    length_epoch = steps_preprocess["length"]
                    overlap = steps_preprocess["overlap"]
                    #---------------------------------------------
                    dist = len(np.arange(tmin, (tmax + overlap) - length_epoch, overlap))

                    X_estim = pipelines[classifier].transform(X_test)

                    X_estim_reshape = X_estim.reshape((-1, dist))
                    X_sum = X_estim_reshape.sum(axis=0)

                    trial_predict = np.where(X_sum < 0, 0, 1)  # if the sum < 0, left. If >0, predict right
                    temporary_accuracy = np.where(trial_predict == Y_test[0::dist-1], 1, 0)  # Compare predictions with observations

                    accuracy[classifier][subject] = temporary_accuracy.mean()

                elif steps_preprocess["score"] == "EAcc":
                    try:
                        accuracy[classifier][subject] = pipelines[classifier].score(X_test, Y_test)
                    except:
                        accuracy[classifier][subject] = np.nan
                else:
                    raise AttributeError("The chosen score does not exist!")

    return accuracy


#################
# LOAD THE DATA #
#################

# Establish master data path
path = os.path.join("Data", "MS")

# Load all files for a given range of participants
# Format: [SUB]_[RUN_NAME].gdf

runNames = ["CE_baseline", "OE_baseline", "R1_acquisition", "R2_acquisition",
            "R3_onlineT", "R4_onlineT", "R5_onlineT", "R6_onlineT"]  # run IDs: CE/OE = closed/open eyes, RX = run #X
subA = ["A" + str(i) for i in range(1, 61)]  # 60 participants (Dataset A ; 29 women ; age 19-59, M = 29, SD = 9.32)
subB = ["B" + str(i) for i in range(61, 82)]  # 21 participants (Dataset B ; 8 women ; age 19-37, M = 29, SD = 9.318)
subC = ["C" + str(i) for i in range(82, 88)]  # 6 additional participants (Dataset C; 4 women; age 20-26, M=22; SD=2.34)
subjects = subA + subB + subC
# Note that all datasets followed the same EEG protocol (MI of left and right hand) with the same channel information

data = {}  # dictionary to hold all our data
for sub in subjects:  # for each subject...
    skip = False
    fnames = [sub + "_" + i + ".gdf" for i in runNames]  # develop a list of all filenames
    sub_data = []  # empty list to hold all of a subject's loaded files
    for f in fnames:  # for each file...
        fpath = os.path.join(path, sub, f)  # select the correct file
        try:
            sub_data.append(mne.io.read_raw_gdf(fpath))  # load it into the list
        except FileNotFoundError:
            skip = True  # if all GDF files are not available, skip this subject (for testing purposes only)
    if not skip:
        # correct channel type information (to properly label EOG and EMG channels)
        new_types = []  # create a new channel types array
        for i in sub_data:
            for j in i.ch_names:
                if "EOG" in j:  # mark ECOG channels
                    new_types.append("ecog")
                elif "EMG" in j:  # mark EMG channels
                    new_types.append("emg")
                else:  # mark the rest as regular EEG
                    new_types.append("eeg")
            i.set_channel_types(dict(zip(i.ch_names, new_types)))  # apply new channel types to raw object
        data[sub] = sub_data  # save sub_data list into data dictionary

# Current data format: data[subject] holds all 8 raw objects
# data[subject][0] = Closed eyes baseline
# data[subject][1] = Open eyes baseline
# data[subject][2] = Training session 1
# data[subject][3] = Training session 2
# data[subject][4] = Test session 1 (w/ feedback)
# data[subject][5] = Test session 2 (w/ feedback)
# data[subject][6] = Test session 3 (w/ feedback)
# data[subject][7] = Test session 4 (w/ feedback)


##################################
# FORMAT DATA FOR INRIA PIPELINE #
##################################

# Follow format used by INRIA pipeline, to speed up analysis
# dic_data_format = participant number (+ _1 or _2 for train) (+ _3-6 for test) : mne raw object
dic_data = {}
for i in data.keys():  # for every subject_session...
    for j in range(1, 7):  # load ALL their data
        session = str(i) + '_' + str(j)  # their numbering will start from 1...
        dic_data[session] = data[i][j + 1]  # but indexing must follow the indexes from the comment above (hence +1)


################################
# SELECT OUR ANALYSIS PIPELINE #
################################

# dictionary for all our testing pipelines
pipelines = {}
pipelines['8csp+lda'] = make_pipeline(CSP(n_components=8), LDA())  # baseline comparison CSP+LDA
pipelines['MDM'] = make_pipeline(Covariances(estimator='lwf'), MDM(metric='riemann', n_jobs=-1))  # simple Riemannian
pipelines['tangentspace+LR'] = make_pipeline(Covariances(estimator='lwf'),
                                             TangentSpace(metric='riemann'),
                                             LogisticRegression())  # more realistic Riemannian


#############################
# RUN OUR ANALYSIS PIPELINE #
#############################

session = list(data.keys())  # a list of participants to be used for analysis
steps_preprocess = {"filter": [8, 30],  # filter from 8-30Hz
                    "drop_channels": ['EOG1', 'EOG2', 'EOG3', 'EMGg', 'EMGd'],  # ignore EOG/EMG channels
                    "tmin": 0.5, "tmax": 4, "overlap": 1/16, "length": 1,
                    "score": "EAcc"}
accuracy = test_pipeline_within_session(pipelines, session, steps_preprocess)  # run pipeline!
