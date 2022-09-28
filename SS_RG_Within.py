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
        train_key = [subject+"_1", subject+"_2", subject+"_3", subject+"_4"]
        test_key = [subject+"_5", subject+"_6", subject+"_7", subject+"_8",
                    subject+"_9", subject+"_10", subject+"_11", subject+"_12"]

        print(subject)
        X_train, Y_train = epoching(dic_data_train, train_key, steps_preprocess)  # X = epochs, Y = labels
        X_test, Y_test = epoching(dic_data_test, test_key, steps_preprocess)  # same as above, but test set

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
path = os.path.join("Data", "SS")

# Load all files for a given range of participants
# Format: [SUB]_[SESS]_[RUN_NAME].gdf

runNames = ["CE_baseline", "OE_baseline",  # CE/OE = closed/open eyes
            "nback1", "nback2", "nback3", "nback4", "nback5", "nback6",
            "R1_acquisition", "R2_acquisition", "R3_acquisition", "R4_acquisition",
            "R5_online", "R6_online", "R7_online", "R8_online", "R9_online", "R10_online", "R11_online", "R12_online",
            "Rspan1", "Rspan2", "Rspan3", "Rspan4", "Rspan5", "Rspan6", "Rspan7", "Rspan8", "Rspan9", "Rspan10"]
sessions = ["S1", "S2", "S3"]  # session IDs
session_folders = ["Session 1", "Session 2", "Session 3"]  # session folders do not line up with session IDs
subjects = next(os.walk(path))[1]  # Note: list of subjects is dynamically assigned here since they are non-consecutive

"""
IMPORTANT! To conform with INRIA pipeline but use multi-session data, each session is appended as a new "subject"
This means that data takes the following format: data[sub_sess], ex: data["S01_S1"] for Subject 1 Session 1
"""


data = {}  # dictionary to hold all our data
for sub in subjects:  # for each subject...
    for sess in range(len(sessions)):  # for each session... (used as an iterator due to sessions/session_folder issue)
        fnames = [sub + "_" + sessions[sess] + "_" + i + ".gdf" for i in runNames]  # develop a list of all filenames
        sub_data = []  # empty list to hold all of a subject's loaded files
        for f in fnames:  # for each file...
            fpath = os.path.join(path, sub, session_folders[sess], f)  # select the correct file
            sub_data.append(mne.io.read_raw_gdf(fpath))  # load it into the list

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
        name = sub + '_' + sessions[sess]  # make a new name so that each session is viewed as its own subject
        data[name] = sub_data  # save sub_data list into data dictionary

# Current data format: data[subject] holds all 30 raw objects
# data[subject_session][0] = Closed eyes baseline
# data[subject_session][1] = Open eyes baseline
# data[subject_session][2] = Nback task 1
# data[subject_session][3] = Nback task 2
# data[subject_session][4] = Nback task 3
# data[subject_session][5] = Nback task 4
# data[subject_session][6] = Nback task 5
# data[subject_session][7] = Nback task 6
# data[subject_session][8] = Training session 1
# data[subject_session][9] = Training session 2
# data[subject_session][10] = Training session 3
# data[subject_session][11] = Training session 4
# data[subject_session][12] = Online session 1 (w/ feedback)
# data[subject_session][13] = Online session 2 (w/ feedback)
# data[subject_session][14] = Online session 3 (w/ feedback)
# data[subject_session][15] = Online session 4 (w/ feedback)
# data[subject_session][16] = Online session 5 (w/ feedback)
# data[subject_session][17] = Online session 6 (w/ feedback)
# data[subject_session][18] = Online session 7 (w/ feedback)
# data[subject_session][19] = Online session 8 (w/ feedback)
# data[subject_session][20] = Reading span task 1
# data[subject_session][21] = Reading span task 2
# data[subject_session][22] = Reading span task 3
# data[subject_session][23] = Reading span task 4
# data[subject_session][24] = Reading span task 5
# data[subject_session][25] = Reading span task 6
# data[subject_session][26] = Reading span task 7
# data[subject_session][27] = Reading span task 8
# data[subject_session][28] = Reading span task 9
# data[subject_session][29] = Reading span task 10


##################################
# FORMAT DATA FOR INRIA PIPELINE #
##################################

# Follow format used by INRIA pipeline, to speed up analysis
# dic_data_format = participant number (+ _1 or _2 for train) (+ _3-6 for test) : mne raw object
dic_data_train = {}
dic_data_test = {}
for i in data.keys():  # for every subject...
    for j in range(1, 5):  # place their training sessions (4) into one dictionary
        session = str(i) + '_' + str(j)  # their numbering will start from 1...
        dic_data_train[session] = data[i][j+7]  # but indices must follow the comment above (hence the +7)
    for j in range(5, 13):  # and their testing sessions (8) into another dictionary
        session = str(i) + '_' + str(j)
        dic_data_test[session] = data[i][j+7]


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
