from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import Covariances
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import os
import mne
import numpy as np
import pandas as pd
from numpy.random import seed
from lime import lime_tabular
mne.set_log_level(verbose="Warning")  # set all the mne verbose to warning

seed(2002012)


class LIMEd():
    """
    This is a dummy pipeline step. Its sole purpose is to take LIME formatted data (n_samples, n_timesteps, n_features)
    and turn it back into MNE formatted data (epochs, features, samples).
    """
    def __init__(self, test):
        self.test = test

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        return LIME_format(X)


class snoop():
    """
    This is a dummy pipeline step. Its sole purpose is to take print the shape of the data at this step
    """
    def __init__(self, test):
        self.test = test

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        print(X.shape)
        return X


def LIME_format(input):
    # reformat data into (n_samples, n_timesteps, n_features)
    return input.transpose(0, 2, 1)


def preprocess(raw, steps={}):
    print("Preprocessing raw data...")  # for progress tracking
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
    print("Epoching the data...")  # for progress tracking
    # we are epoching for the RG classifiers
    """From the dictionary of mne.rawGDF extract all the epochs selected with Key_session
     Return the epochs list as X and the label as Y"""

    # ---------------------------------------------
    tmin = steps_preprocess["tmin"]
    tmax = steps_preprocess["tmax"]
    length_epoch = steps_preprocess["length"]
    overlap = steps_preprocess["overlap"]
    # ---------------------------------------------
    X = None
    Y = None

    for key in key_session:
        if steps_preprocess is not None:
            _ = preprocess(dict[key], steps_preprocess)

        # MNE recommends the following process prior to signal decimation:
        current_sfreq = dict[key].info['sfreq']
        desired_sfreq = 256  # Hz
        decim = np.round(current_sfreq / desired_sfreq).astype(int)
        obtained_sfreq = current_sfreq / decim
        lowpass_freq = obtained_sfreq / 3.0
        dict[key].filter(l_freq=None, h_freq=lowpass_freq, n_jobs=-1)

        epoch = mne.Epochs(dict[key], mne.events_from_annotations(dict[key], key_events)[0], tmin=-1, tmax=5,
                           baseline=(None, 0), verbose="CRITICAL", decim=decim)[list(key_events.values())]
        # NOTE: we are decimating the signal to 256 Hz and only grabbing 2 events to speed up processing

        list_start = np.arange(tmin, (tmax + overlap) - length_epoch, overlap)
        list_stop = np.arange(tmin + length_epoch, (tmax + overlap), overlap)

        for start, stop in zip(list_start, list_stop):
            if X is None:
                X = epoch.get_data(tmin=start, tmax=stop)
                Y = epoch.events[:, 2]

            else:
                X = np.append(X, epoch.get_data(tmin=start, tmax=stop), axis=0)
                Y = np.append(Y, epoch.events[:, 2], axis=0)

    return X, Y


def test_pipeline(test, pipelines, session, steps_preprocess):
    """ Take in input the different pipelines to test and return the corresponding classification accuracy"""
    accuracy = pd.DataFrame(np.zeros((len(session), len(pipelines))), index=session, columns=pipelines.keys())

    for subject in session:  # this will be something like "A10" for MS and "S06_S1" for SS
        print("Running a leave-one-out classification, leaving out {}".format(subject))  # for progress tracking
        train_key = {k: v for k, v in test.items() if subject not in k}  # train on all subjects data EXCEPT one
        test_key = {k: v for k, v in test.items() if subject in k}  # test on that subject

        X_train, Y_train = epoching(test, train_key, steps_preprocess)  # X = epochs, Y = labels
        X_test, Y_test = epoching(test, test_key, steps_preprocess)  # same as above, but test set

        for classifier in pipelines.keys():
            print("Fitting classifier: {}".format(classifier))
            newTrain = LIME_format(X_train.copy())
            newTest = LIME_format(X_test.copy())
            # fit must get LIMEd data, since it will automatically un-lime it during training
            pipelines[classifier].fit(newTrain, Y_train)

            if steps_preprocess["score"] == "TAcc":

                # ---------------------------------------------
                tmin = steps_preprocess["tmin"]
                tmax = steps_preprocess["tmax"]
                length_epoch = steps_preprocess["length"]
                overlap = steps_preprocess["overlap"]
                # ---------------------------------------------
                dist = len(np.arange(tmin, (tmax + overlap) - length_epoch, overlap))

                X_estim = pipelines[classifier].transform(X_test)

                X_estim_reshape = X_estim.reshape((-1, dist))
                X_sum = X_estim_reshape.sum(axis=0)

                trial_predict = np.where(X_sum < 0, 0, 1)  # if the sum < 0, left. If >0, predict right
                temporary_accuracy = np.where(trial_predict == Y_test[0::dist - 1], 1, 0)  # Compare predictions with observations

                accuracy.loc[subject, classifier] = temporary_accuracy.mean()

            elif steps_preprocess["score"] == "EAcc":
                try:
                    accuracy.loc[subject, classifier] = pipelines[classifier].score(X_test, Y_test)
                except:
                    accuracy.loc[subject, classifier] = np.nan
            elif steps_preprocess["score"] == "LIME":
                print("Caclulating LIME values...")
                channels = ['Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'C1', 'C3', 'C5', 'C2', 'C4',
                            'C6', 'F4', 'FC2', 'FC4', 'FC6', 'CP2', 'CP4', 'CP6', 'P4',
                            'F3', 'FC1', 'FC3', 'FC5', 'CP1', 'CP3', 'CP5', 'P3']
                explainer = lime_tabular.RecurrentTabularExplainer(newTrain, training_labels=Y_train,
                                                                   feature_names=channels,
                                                                   discretize_continuous=True,
                                                                   class_names=['Left', 'Right'],
                                                                   discretizer='decile')

                # TODO: make this compute average of every sample with a given label, rather than a random sample
                N = np.random.randint(0, 492)  # grab a random epoch
                exp = explainer.explain_instance(newTest[N], pipelines[classifier].predict_proba,
                                                 num_features=27, labels=(0, 1))  # explain it

                accuracy.loc[subject, classifier] = exp  # save the explanation
            else:
                raise AttributeError("The chosen score does not exist!")

    return accuracy


def load_MS(between=False, within=False):
    print("Loading the MS dataset (all subjects that are available)...")  # for progress tracking
    # Establish master data path
    path = os.path.join("Data", "MS")

    # Load all files for a given range of participants
    # Format: [SUB]_[RUN_NAME].gdf

    runNames = ["CE_baseline", "OE_baseline", "R1_acquisition", "R2_acquisition",
                "R3_onlineT", "R4_onlineT", "R5_onlineT",
                "R6_onlineT"]  # run IDs: CE/OE = closed/open eyes, RX = run #X
    subA = ["A" + str(i) for i in range(1, 61)]  # 60 participants (Dataset A ; 29 women ; age 19-59, M = 29, SD = 9.32)
    subB = ["B" + str(i) for i in
            range(61, 82)]  # 21 participants (Dataset B ; 8 women ; age 19-37, M = 29, SD = 9.318)
    subC = ["C" + str(i) for i in
            range(82, 88)]  # 6 additional participants (Dataset C; 4 women; age 20-26, M=22; SD=2.34)
    subjects = subA + subB + subC
    # Note that all data followed the same EEG protocol (MI of left and right hand) with the same channel information

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

    if between:
        # dic_data_format = participant number (+ _N for train) (+ _N++ for test) : mne raw object
        dic_data = {}
        for i in data.keys():  # for every subject...
            for j in range(1, 7):
                session = str(i) + '_' + str(j)
                dic_data[session] = data[i][j + 1]  # place their sessions into one dictionary following indexes above
        return data, dic_data
    if within:
        dic_data_train = {}
        dic_data_test = {}
        for i in data.keys():  # for every subject...
            for j in range(1, 3):  # place their training sessions into one dictionary
                session = str(i) + '_' + str(j)
                dic_data_train[session] = data[i][j + 1]  # following the indexes from the comment above
            for j in range(3, 7):  # and their testing sessions into another dictionary
                session = str(i) + '_' + str(j)
                dic_data_test[session] = data[i][j + 1]  # with the same indexing as the comment block above
        return data, dic_data_train, dic_data_test


def MS_RG_Between():
    # load the MS dataset
    data, dic_data = load_MS(between=True)

    # run the selected pipelines
    session = list(data.keys())  # a list of participants to be used for analysis
    steps_preprocess = {"filter": [8, 30],  # filter from 8-30Hz
                        "drop_channels": ['EOG1', 'EOG2', 'EOG3', 'EMGg', 'EMGd'],  # ignore EOG/EMG channels
                        "tmin": 0.5, "tmax": 4, "overlap": 1/16, "length": 1,
                        "score": "LIME"}
    accuracy = test_pipeline(dic_data, pipelines, session, steps_preprocess)  # run it!

    return accuracy


# dictionary for all our testing pipelines
pipelines = {}
pipelines['8csp+lda'] = make_pipeline(LIMEd(test=True),
                                      CSP(n_components=8),
                                      LDA())  # baseline comparison CSP+LDA
pipelines['MDM'] = make_pipeline(LIMEd(test=True),
                                 Covariances(estimator='lwf'),
                                 snoop(test=True),
                                 MDM(metric='riemann', n_jobs=-1))  # simple Riemannian
pipelines['tangentspace+LR'] = make_pipeline(LIMEd(test=True),
                                             Covariances(estimator='lwf'),
                                             TangentSpace(metric='riemann'),
                                             LogisticRegression(max_iter=350, n_jobs=-1))  # more realistic Riemannian

