import shap
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
from tqdm import tqdm
mne.set_log_level(verbose="Warning")  # set all the mne verbose to warning
sfreq = 0
ch_names = []
seed(2002012)


class SHAPed():
    """
    This is a dummy pipeline step. Its sole purpose is to take SHAP formatted data (n_samples, n_timesteps*n_features)
    and turn it back into MNE formatted data (epochs, features, samples).
    """
    def __init__(self, test):
        self.test = test

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        return SHAP_format(X)


def SHAP_format(input):
    if len(input.shape) == 3:  # if 3D...
        # reformat data into 2D shape that SHAP expects
        epochs = input.shape[0]
        channels = input.shape[1]
        timsteps = input.shape[2]
        return np.reshape(input, (epochs, channels*timsteps))
    elif len(input.shape) == 2:  # if 2D...
        # reformat data into 3D shape that MNE expects
        epochs = input.shape[0]
        channel_steps = input.shape[1]
        global sfreq
        return np.reshape(input, (epochs, channel_steps//sfreq, sfreq))


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

        epoch = mne.Epochs(dict[key], mne.events_from_annotations(dict[key], key_events)[0], tmin=-1, tmax=5,
                           baseline=(None, 0), verbose="CRITICAL")[list(key_events.values())]

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
    accuracy = []

    for subject in session:  # this will be something like "A10" for MS and "S06_S1" for SS
        print("Running a leave-one-out classification, leaving out {}".format(subject))  # for progress tracking
        train_key = {k: v for k, v in test.items() if subject not in k}  # train on all subjects data EXCEPT one
        test_key = {k: v for k, v in test.items() if subject in k}  # test on that subject

        X_train, Y_train = epoching(test, train_key, steps_preprocess)  # X = epochs, Y = labels
        X_test, Y_test = epoching(test, test_key, steps_preprocess)  # same as above, but test set

        for classifier in pipelines.keys():
            print("Fitting classifier: {}".format(classifier))
            scale = lambda x: x * 1e6  # function to scale EEG data
            newTrain = np.apply_along_axis(scale, 1, SHAP_format(X_train.copy()))  # scale the samples, but
            newTest = np.apply_along_axis(scale, 1, SHAP_format(X_test.copy()))  # only apply it to the timesteps dim

            if steps_preprocess["score"] == "TAcc":
                # fit gets un-SHAPed data
                pipelines[classifier].fit(X_train, Y_train)

                print("Calculating classifier accuracy...")
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

                accuracy.append(
                    dict(
                        subject=subject,
                        classifier=classifier,
                        value=temporary_accuracy.mean()
                    )
                )

            elif steps_preprocess["score"] == "EAcc":
                # fit gets un-SHAPed data
                pipelines[classifier].fit(X_train, Y_train)

                print("Calculating classifier accuracy...")
                try:
                    accuracy.append(
                        dict(
                            subject=subject,
                            classifier=classifier,
                            value=pipelines[classifier].score(X_test, Y_test)
                        )
                    )
                except:
                    accuracy.append(
                        dict(
                            subject=subject,
                            classifier=classifier,
                            value=np.nan
                        )
                    )
            elif steps_preprocess["score"] == "SHAP":
                # fit must get SHAPed data, since it will automatically un-SHAP it during training
                pipelines[classifier].fit(newTrain, Y_train)

                print("Calculating SHAP values...")
                global sfreq  # grab our global sampling frequency
                global ch_names  # grab our global channel names
                channels = []
                for c in ch_names:
                    for i in range(sfreq):
                        name = c + "_t-" + str(i)
                        channels.append(name)  # convert labels from channels to channelsteps

                explainer = shap.KernelExplainer(pipelines[classifier].predict, newTrain)

                SHAPmeans = {feature: [] for feature in channels}  # dictionary for predictions of both possible classes
                for instance in tqdm(range(len(newTest))):  # for each epoch...
                    shap_values = explainer.shap_values(newTest[instance, :], namples=10)

                    for i in range(len(channels)):  # for each Channel_t-timestamp's contribution
                        feature = channels[i]
                        value = shap_values[i]
                        SHAPmeans[feature].append(value)

                print("Averaging SHAP values...")
                for feature in channels:  # for each Channel_t-timestamp...
                    # calculate the average contribution for each feature
                    SHAPmeans[feature] = np.mean(SHAPmeans[feature])

                accuracy.append(
                    dict(
                        subject=subject,
                        classifier=classifier,
                        value=SHAPmeans
                    )
                )
            else:
                raise AttributeError("The chosen score does not exist!")

        # now that we have completed 1 subject, save their data to a csv file

        # create dataframe for storage
        colnames = ["Classifier", "Predicted", "Channel", "Time", "Weight"]
        df = []
        for SHAP in accuracy:  # load each LIME dictionary from our results array
            # grab relevant information
            subject = SHAP["subject"]  # string
            classifier = SHAP["classifier"]  # string
            data = SHAP["value"]  # dictionary {string: float}

            for i in data.keys():  # for each channel-step
                key = i
                value = data[i]

                channel = key.split("-")[0][:-2]  # grab channel name, remove "_t" from end (string, 10-20 system)
                time = key.split("-")[1]  # grab timestamp (int, 0-511 or 0-255 depending on sampling rate)
                weight = value  # grab values

                row = [classifier, i, channel, time, weight]
                df.append(row)

        out = pd.DataFrame(df, columns=colnames)
        filename = subject + ".csv"
        out.to_csv(filename)


def load_MS(between=False, within=False):
    print("Loading the MS dataset (all subjects that are available)...")  # for progress tracking
    # Establish master data path
    path = os.path.join("../Data", "MS")

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

            global sfreq
            sfreq = int(sub_data[0].info["sfreq"])
            global ch_names
            ch_names = sub_data[0].info["ch_names"]
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
                        "score": "SHAP"}
    test_pipeline(dic_data, pipelines, session, steps_preprocess)  # run it!

# dictionary for all our testing pipelines
pipelines = {}
#pipelines['8csp+lda'] = make_pipeline(SHAPed(test=True),
#                                      CSP(n_components=8),
#                                      LDA())  # baseline comparison CSP+LDA
pipelines['MDM'] = make_pipeline(SHAPed(test=True),
                                 Covariances(estimator='lwf'),
                                 MDM(metric='riemann', n_jobs=-1))  # simple Riemannian
pipelines['tangentspace+LR'] = make_pipeline(SHAPed(test=True),
                                             Covariances(estimator='lwf'),
                                             TangentSpace(metric='riemann'),
                                             LogisticRegression(max_iter=350, n_jobs=-1))  # more realistic Riemannian
