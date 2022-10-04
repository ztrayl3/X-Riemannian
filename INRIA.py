import os
import mne
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from tensorflow import one_hot


def create_key(df, train=1, test=1):  # Create an array of keys to indicate train/test datasets
    if "A59" in df:  # handle potential problems from subject A59
        keys = np.empty(shape=len(df) * train * 2 + (len(df) * test * 4) - 2 * test,  dtype=np.dtype('<U8'))
    else:
        keys = np.empty(shape=len(df) * train * 2 + (len(df) * test * 4),  dtype=np.dtype('<U8'))
    i = 0
    for val in df:
        if train:
            keys[i: i+2] = [val+"_1", val+"_2"]
            i += 2
        if test:
            if val == "A59":
                keys[i: i+2] = [val+"_3", val+"_4"]
                i += 2
            else:
                keys[i: i+4] = [val+"_3", val+"_4",  val+"_5", val+"_6"]
                i += 4

    return keys


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


def epoching(dict, key_session=[], steps_preprocess=None, key_events={"769":0, "770": 1}, DL=False):
    if DL:  # if we are epoching for the Deep Learning classifier...
        """From the dictionary of mne.rawGDF extract all the epochs selected with Key_session
         Return the epochs list as X and tje label as Y"""

        #---------------------------------------------
        tmin= steps_preprocess["tmin"]
        tmax = steps_preprocess["tmax"]
        length_epoch = steps_preprocess["length"]
        overlap = steps_preprocess["overlap"]
        #---------------------------------------------

        list_start = np.arange(tmin, (tmax + overlap) - length_epoch, overlap)
        list_stop = np.arange(tmin+length_epoch, (tmax+overlap), overlap)

        n_chans = dict[key_session[0]].get_channel_types().count("eeg")  # take the first EEG file, how many EEG channels?
        time_step = int(length_epoch * dict[key_session[0]].info['sfreq'])  # get sampling frequency
        n_events = len(list_start) * 40 * len(key_session)  # 40 represent the number of events in each raw data

        X = np.zeros((n_events, n_chans, time_step))  # build properly shaped input arrays
        Y = np.zeros(n_events)

        i = 0

        for key in key_session:
            if steps_preprocess is not None:
                _ = preprocess(dict[key], steps_preprocess)  # preprocess data

            epoch = mne.Epochs(dict[key], mne.events_from_annotations(dict[key], key_events)[0],
                               tmin=-1, tmax=5)  # epoch data from annotations

            assert len(epoch.events[:, 2]) == 40, ("'%s' don't have 40 events it actually have %s " %
                                                   (key, len(epoch.events[:, 2])))

            for start, stop in zip(list_start, list_stop):

                X[i: i + 40] = epoch.get_data(tmin=start, tmax=stop)
                Y[i: i + 40] = epoch.events[:, 2]
                i += 40

        Y = one_hot(Y, depth=2)  # one-hot encode the labels

        return X, Y

    else:  # we are epoching for the RG classifiers
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
                               baseline=(None, 0), verbose="CRITICAL")

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


class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


def test_pipeline(test, pipelines, session, steps_preprocess, train=None,
                  between=False, within=False, MS=False, SS=False):
    """ Take in input the different pipelines to test and return the corresponding classification accuracy"""
    accuracy = pd.DataFrame(np.zeros((len(session), len(pipelines))), index=session, columns=pipelines.keys())

    for subject in session:  # this will be something like "A10" for MS and "S06_S1" for SS
        if between:  # between subject/session analysis: no train set, just leave-one-out
            if MS:  # using the multi subject dataset
                train_key = {k: v for k, v in test.items() if subject not in k}  # train on all subjects data EXCEPT one
                test_key = {k: v for k, v in test.items() if subject in k}  # test on that subject

                X_train, Y_train = epoching(test, train_key, steps_preprocess)  # X = epochs, Y = labels
                X_test, Y_test = epoching(test, test_key, steps_preprocess)  # same as above, but test set

            if SS:  # using the single subject dataset
                # group by subject, since we're focused on between-session not subject here
                # We will take the "subject" (in this case S0#_S#) and treat that as our leave-out session
                subID = subject[:3]  # grab the subject ID
                subset = {k: v for k, v in test.items() if subID in k}  # subset to just the subject of interest
                train_key = {k: v for k, v in subset.items() if subject not in k}  # train w/all sessions EXCEPT one (2)
                test_key = {k: v for k, v in subset.items() if subject in k}  # test on that session (1)

                X_train, Y_train = epoching(subset, train_key, steps_preprocess)  # X = epochs, Y = labels
                X_test, Y_test = epoching(subset, test_key, steps_preprocess)  # same as above, but test set

        if within:  # within subject/session analysis: utilize the built-in train/test sets
            if MS:  # using the multi subject dataset
                train_key = [subject + "_1", subject + "_2"]
                if subject == "A59":  # Manage the error during the data acquisition of A59
                    test_key = [subject + "_3", subject + "_4"]
                else:  # Take all the session possible
                    test_key = [subject + "_3", subject + "_4", subject + "_5", subject + "_6"]

                X_train, Y_train = epoching(train, train_key, steps_preprocess)  # X = epochs, Y = labels
                X_test, Y_test = epoching(test, test_key, steps_preprocess)  # same as above, but test set

            if SS:  # using the single subject dataset
                train_key = [subject + "_1", subject + "_2", subject + "_3", subject + "_4"]
                test_key = [subject + "_5", subject + "_6", subject + "_7", subject + "_8",
                            subject + "_9", subject + "_10", subject + "_11", subject + "_12"]

                X_train, Y_train = epoching(train, train_key, steps_preprocess)  # X = epochs, Y = labels
                X_test, Y_test = epoching(test, test_key, steps_preprocess)  # same as above, but test set

        for classifier in pipelines.keys():
            pipelines[classifier].fit(X_train, Y_train)

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
                temporary_accuracy = np.where(trial_predict == Y_test[0::dist - 1], 1,
                                              0)  # Compare predictions with observations

                accuracy[classifier][subject] = temporary_accuracy.mean()

            elif steps_preprocess["score"] == "EAcc":
                try:
                    accuracy[classifier][subject] = pipelines[classifier].score(X_test, Y_test)
                except:
                    accuracy[classifier][subject] = np.nan
            else:
                raise AttributeError("The chosen score does not exist!")

    return accuracy


def load_MS(between=False, within=False):
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


def load_SS(between=False, within=False):
    # Establish master data path
    path = os.path.join("Data", "SS")

    # Load all files for a given range of participants
    # Format: [SUB]_[SESS]_[RUN_NAME].gdf

    runNames = ["CE_baseline", "OE_baseline",  # CE/OE = closed/open eyes
                "nback1", "nback2", "nback3", "nback4", "nback5", "nback6",
                "R1_acquisition", "R2_acquisition", "R3_acquisition", "R4_acquisition",
                "R5_online", "R6_online", "R7_online", "R8_online", "R9_online", "R10_online", "R11_online",
                "R12_online",
                "Rspan1", "Rspan2", "Rspan3", "Rspan4", "Rspan5", "Rspan6", "Rspan7", "Rspan8", "Rspan9", "Rspan10"]
    sessions = ["S1", "S2", "S3"]  # session IDs
    session_folders = ["Session 1", "Session 2", "Session 3"]  # session folders do not line up with session IDs
    subjects = next(os.walk(path))[
        1]  # Note: list of subjects is dynamically assigned here since they are non-consecutive

    data = {}  # dictionary to hold all our data
    for sub in subjects:  # for each subject...
        data[sub] = [[] for i in range(len(sessions))]  # within subject, create a list for each session
        for sess in range(
                len(sessions)):  # for each session... (used as an iterator due to sessions/session_folder issue)
            fnames = [sub + "_" + sessions[sess] + "_" + i + ".gdf" for i in
                      runNames]  # develop a list of all filenames
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
            data[sub][sess] = sub_data  # save sub_data list into data dictionary

    # Current data format: data[subject][session] holds all 30 raw objects for a given subject's session
    # data[subject][session][0] = Closed eyes baseline
    # data[subject][session][1] = Open eyes baseline
    # data[subject][session][2] = Nback task 1
    # data[subject][session][3] = Nback task 2
    # data[subject][session][4] = Nback task 3
    # data[subject][session][5] = Nback task 4
    # data[subject][session][6] = Nback task 5
    # data[subject][session][7] = Nback task 6
    # data[subject][session][8] = Training session 1
    # data[subject][session][9] = Training session 2
    # data[subject][session][10] = Training session 3
    # data[subject][session][11] = Training session 4
    # data[subject][session][12] = Online session 1 (w/ feedback)
    # data[subject][session][13] = Online session 2 (w/ feedback)
    # data[subject][session][14] = Online session 3 (w/ feedback)
    # data[subject][session][15] = Online session 4 (w/ feedback)
    # data[subject][session][16] = Online session 5 (w/ feedback)
    # data[subject][session][17] = Online session 6 (w/ feedback)
    # data[subject][session][18] = Online session 7 (w/ feedback)
    # data[subject][session][19] = Online session 8 (w/ feedback)
    # data[subject][session][20] = Reading span task 1
    # data[subject][session][21] = Reading span task 2
    # data[subject][session][22] = Reading span task 3
    # data[subject][session][23] = Reading span task 4
    # data[subject][session][24] = Reading span task 5
    # data[subject][session][25] = Reading span task 6
    # data[subject][session][26] = Reading span task 7
    # data[subject][session][27] = Reading span task 8
    # data[subject][session][28] = Reading span task 9
    # data[subject][session][29] = Reading span task 10

    if between:
        # dic_data_format = participant number (+ _N for train) (+ _N++ for test) : mne raw object
        dic_data = {}
        for sub in data.keys():  # for every subject...
            for sess in range(len(sessions)):  # for each session...
                for i in range(1, 13):  # place all their relevant train/test sessions into one dictionary
                    session = sub + "_" + sessions[sess] + "_" + str(i)  # their numbering will start from 1...
                    dic_data[session] = data[sub][sess][i + 7]  # but indices must follow above rules (hence the +7)
        return data, dic_data
    if within:
        dic_data_train = {}
        dic_data_test = {}
        for sub in data.keys():  # for every subject...
            for sess in range(len(sessions)):  # for each session...
                for i in range(1, 5):  # place their training sessions (4) into one dictionary
                    session = sub + "_" + sessions[sess] + "_" + str(i)
                    dic_data_train[session] = data[sub][sess][i + 7]
                for i in range(5, 13):  # and their testing sessions (8) into another dictionary
                    session = sub + "_" + sessions[sess] + "_" + str(i)
                    dic_data_test[session] = data[sub][sess][i + 7]
        return data, dic_data_train, dic_data_test

