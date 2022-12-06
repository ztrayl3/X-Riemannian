import os
import mne
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.random import seed
from lime import lime_tabular
from tensorflow import one_hot
from keras.models import load_model
from sklearn.pipeline import make_pipeline
from tensorflow.keras.utils import Sequence
from EEGModels import ShallowConvNet, square, log
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

mne.set_log_level(verbose="Warning")  # set all the mne verbose to warning
seed(2002012)
set_random_seed(2002012)


class LIMEd():
    """
    This is a dummy pipeline step. Its sole purpose is to take LIME formatted data (n_samples, n_timesteps, n_features)
    and turn it back into MNE formatted data (epochs, features, samples).
    """
    def __init__(self, DL=False):
        self.DL = DL

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        if self.DL:  # perform DL specific transpose
            X.x = LIME_format(X.x)
            return X
        return LIME_format(X)


class LIMEdl():
    """
    This is a dummy pipeline step. Its sole purpose is to fit the functions for a neural network into an SKlearn
    pipeline. Note that this can be done with KerasRegressor as well, but this is simpler for my simpler needs.
    """
    def __init__(self, channels, valid_gen):
        self.model = ShallowConvNet(nb_classes=2, Chans=channels, Samples=1024)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=BinaryAccuracy())
        self.batch_size = 32
        self.n_epochs = 500
        self.callbacks = [EarlyStopping(patience=50, monitor="loss"),
                          ModelCheckpoint(filepath='Model/best.h5', save_best_only=True)]
        self.valid_gen = valid_gen

    def fit(self, X, y=None, **fit_params):
        self.model.fit(x=X,
                       y=y,
                       batch_size=self.batch_size,
                       epochs=self.n_epochs,
                       callbacks=self.callbacks,
                       validation_data=self.valid_gen,
                       verbose=1)
        return self.model

    def predict(self, X):
        self.model = load_model("Model/best.h5", custom_objects={"square": square, "log": log})
        return self.model.predict(x=X)

    def score(self, X, y=None, sample_weight=None):
        self.model = load_model("Model/best.h5", custom_objects={"square": square, "log": log})
        return self.model.evaluate(x=X, y=y)[1]


def LIME_format(input):
    # reformat data into (n_samples, n_timesteps, n_features)
    return input.transpose(0, 2, 1)


def LIME_calc(Xtrain, Ytrain, Xtest, labels, predictor, sfreq):
    print("Calculating LIME values...")
    explainer = lime_tabular.RecurrentTabularExplainer(Xtrain, training_labels=Ytrain,
                                                       feature_names=labels,
                                                       discretize_continuous=False,
                                                       class_names=['Left', 'Right'])
    """
                                                         mimics    [   0       1   ]
    """

    first = True
    LIMEans = dict(Left={}, Right={})  # dictionary for predictions of both possible classes
    # for instance in tqdm(range(len(Xtest))):  # for each epoch...
    for instance in tqdm(range(50)):  # for DEBUGGING
        for i in range(0, 2):  # explain for one class at a time
            sample = Xtest[instance]
            exp = explainer.explain_instance(sample, predictor,
                                             num_features=int(len(channels) * sfreq),
                                             num_samples=1000,
                                             labels=(i,))  # choose the correct class

            if first:  # only create dictionary once
                # for each Channel_t-timestamp, give them an empty list in a dict
                LIMEans["Left"] = {feature: [] for feature in exp.domain_mapper.feature_names}
                LIMEans["Right"] = {feature: [] for feature in exp.domain_mapper.feature_names}
                first = False

            for feature in exp.local_exp[i]:  # for each Channel_t-timestamp's contribution
                index = feature[0]
                val = feature[1]
                if i == 0:
                    LIMEans["Left"][exp.domain_mapper.feature_names[index]].append(val)  # save it in the dict
                elif i == 1:
                    LIMEans["Right"][exp.domain_mapper.feature_names[index]].append(val)  # save it in the dict

    print("Averaging LIME values...")
    for feature in exp.domain_mapper.feature_names:  # for each Channel_t-timestamp...
        # calculate the average contribution for each feature (and class)
        LIMEans["Left"][feature] = np.mean(LIMEans["Left"][feature])
        LIMEans["Right"][feature] = np.mean(LIMEans["Right"][feature])

    return LIMEans


def LIME_output(data):
    # create dataframe for storage
    classes = ["left", "right"]
    colnames = ["Subject", "Session", "Classifier", "Condition", "Predicted", "Accuracy", "Channel", "Time", "Weight"]
    df = []
    for LIME in data:  # load each LIME dictionary from our results array
        # grab relevant information
        subject = LIME["subject"]  # string
        session = LIME["sess"]  # string
        classifier = LIME["classifier"]  # string
        condition = LIME["condition"]  # string
        acc = LIME["acc"]  # float
        LEFT = LIME["value"]["Left"]  # dictionary {string: float}
        RIGHT = LIME["value"]["Right"]  # dictionary {string: float}

        for i in classes:  # for left / right predictions...
            for j in LEFT.keys():  # for each channel-step
                key = j
                if i == "left":
                    value = LEFT[j]
                if i == "right":
                    value = RIGHT[j]

                channel = key.split("-")[0][:-2]  # grab channel name, remove "_t" from end (string, 10-20 system)
                t = key.split("-")[1]  # grab timestamp (int, 0-511 or 0-255 depending on sampling rate)
                weight = value  # grab values

                # "Subject", "Session", "Classifier", "Condition", "Predicted", "Accuracy", "Channel", "Time", "Weight"
                row = [subject, session, classifier, condition, i, acc, channel, t, weight]
                df.append(row)

    out = pd.DataFrame(df, columns=colnames)
    path = "Stats/"
    # mark file with a timestamp, just to ensure uniqueness in filename
    filename = path + subject + "_" + str(time.time()).replace(".", "") + ".csv"
    out.to_csv(filename)


def create_key_MS(df, train=1, test=1):  # Create an array of keys to indicate train/test datasets
    print("Creating keys...")  # for progress tracking
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


def create_key_SS(df, train=1, test=1):  # Create an array of keys to indicate train/test datasets
    print("Creating keys...")  # for progress tracking
    keys = np.empty(shape=len(df) * train * 4 + (len(df) * test * 8), dtype=np.dtype('<U8'))
    i = 0
    for val in df:
        if train:
            keys[i: i+4] = [val+"_1", val+"_2", val+"_3", val+"_4"]
            i += 4
        if test:
            keys[i: i+8] = [val+"_5", val+"_6", val+"_7", val+"_8", val+"_9", val+"_10", val+"_11", val+"_12"]
            i += 8

    return keys


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


def epoching(dict, key_session=[], steps_preprocess=None, key_events={"769": 0, "770": 1}, DL=False):
    print("Epoching the data...")  # for progress tracking
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

        n_chans = dict[key_session[0]].get_channel_types().count("eeg")  # take the first EEG file, how many channels?
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

            event_count = len(epoch.events[:, 2])

            for start, stop in zip(list_start, list_stop):

                X[i: i + event_count] = epoch.get_data(tmin=start, tmax=stop)
                Y[i: i + event_count] = epoch.events[:, 2]
                i += event_count

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


class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shape = self.x.shape

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


def test_pipeline(test, pipelines, session, steps_preprocess, train=None,
                  between=False, within=False, MS=False, SS=False, subs=None):
    """ Take in input the different pipelines to test and return the corresponding classification accuracy"""

    if not subs:  # if no subject's provided
        subs = session  # use all subjects available

    for subject in session:  # this will be something like "A10" for MS and "S06_S1" for SS
        if subject not in subs:  # skip this loop if it's not one of the subjects we want
            continue

        # variables for eventual output
        subID = subject
        label = "NA"
        if between:
            cond = "B"  # between subjects
        else:
            cond = "W"  # within subjects
        accuracy = []  # reset the results array for each subject to keep files from concatenating
        if between:  # between subject/session analysis: no train set, just leave-one-out
            if MS:  # using the multi subject dataset
                print("Running a leave-one-out classification, leaving out {}".format(subject))  # for progress tracking
                train_key = {k: v for k, v in test.items() if subject not in k}  # train on all subjects data EXCEPT one
                test_key = {k: v for k, v in test.items() if subject in k}  # test on that subject

                X_train, Y_train = epoching(test, train_key, steps_preprocess)  # X = epochs, Y = labels
                X_test, Y_test = epoching(test, test_key, steps_preprocess)  # same as above, but test set

            if SS:  # using the single subject dataset
                print("Running a leave-one-out classification grouped by subject, leaving out {}".format(subject))  # for progress tracking
                # group by subject, since we're focused on between-session not subject here
                # We will take the "subject" (in this case S0#_S#) and treat that as our leave-out session
                subID = subject[:3]  # grab the subject ID
                label = subject[-3:]  # grab the session ID
                subset = {k: v for k, v in test.items() if subID in k}  # subset to just the subject of interest
                train_key = {k: v for k, v in subset.items() if subject not in k}  # train w/all sessions EXCEPT one (2)
                test_key = {k: v for k, v in subset.items() if subject in k}  # test on that session (1)

                X_train, Y_train = epoching(subset, train_key, steps_preprocess)  # X = epochs, Y = labels
                X_test, Y_test = epoching(subset, test_key, steps_preprocess)  # same as above, but test set

        if within:  # within subject/session analysis: utilize the built-in train/test sets
            if MS:  # using the multi subject dataset
                print("Running a simple train/test set classification with each MS subject"
                      " trained and tested on their own data. Running subject {}".format(subject))   # for progress tracking
                train_key = [subject + "_1", subject + "_2"]
                if subject == "A59":  # Manage the error during the data acquisition of A59
                    test_key = [subject + "_3", subject + "_4"]
                else:  # Take all the session possible
                    test_key = [subject + "_3", subject + "_4", subject + "_5", subject + "_6"]

                X_train, Y_train = epoching(train, train_key, steps_preprocess)  # X = epochs, Y = labels
                X_test, Y_test = epoching(test, test_key, steps_preprocess)  # same as above, but test set

            if SS:  # using the single subject dataset
                print("Running a simple train/test set classification with each SS subject"
                      " trained and tested on their own data. Running subject {}".format(subject))  # for progress tracking
                subID = subject[:3]  # grab the subject ID
                label = subject[-3:]  # grab the session ID
                train_key = [subject + "_1", subject + "_2", subject + "_3", subject + "_4"]
                test_key = [subject + "_5", subject + "_6", subject + "_7", subject + "_8",
                            subject + "_9", subject + "_10", subject + "_11", subject + "_12"]

                X_train, Y_train = epoching(train, train_key, steps_preprocess)  # X = epochs, Y = labels
                X_test, Y_test = epoching(test, test_key, steps_preprocess)  # same as above, but test set

        for classifier in pipelines.keys():
            print("Fitting classifier: {}".format(classifier))
            scale = lambda x: x * 1e6  # function to scale EEG data
            newTrain = np.apply_along_axis(scale, 1, LIME_format(X_train.copy()))  # scale the samples, but
            newTest = np.apply_along_axis(scale, 1, LIME_format(X_test.copy()))  # only apply it to the timesteps dim

            # fit must get LIMEd data, since it will automatically un-lime it during training
            pipelines[classifier].fit(newTrain, Y_train)
            score = pipelines[classifier].score(newTest, Y_test)  # grab classification accuracy

            global channels  # grab our global channel names
            global sfreq  # grab our global sampling frequency

            accuracy.append(
                dict(
                    subject=subID,
                    sess=label,
                    classifier=classifier,
                    condition=cond,
                    acc=score,
                    value=LIME_calc(newTrain, Y_train, newTest, channels,
                                    pipelines[classifier].predict_proba, sfreq)
                )
            )

        # now that we have completed 1 subject (3 classifiers), save their data to a csv file
        LIME_output(accuracy)


def test_pipeline_DL(data, dic_data_train, steps_preprocess, dic_data_test=None,
                     between=False, within=False, MS=False, SS=False, subs=None):
    global channels
    subjects = list(data.keys())  # a list of participants to be used for analysis
    cond = "NA"

    if not subs:  # if no subject's provided
        subs = subjects  # use all subjects available

    for subject in (range(len(subjects))):  # for each subject
        ID = subjects[subject]  # get string ID
        if ID not in subs:  # skip this loop if it's not one of the subjects we want
            continue
        print("Targeting subject {}".format(ID))
        accuracy = []  # reset the results array for each subject to keep files from concatenating
        if MS:
            if between:
                print("Running a leave-one-out classification, leaving out {}".format(ID))  # for progress tracking
                cond = "B"
                train_id = subjects.copy()  # train on all
                test_id = train_id.pop(subject)  # except for one (leave one out)

                key_train_valid = create_key_MS(train_id, train=1, test=1)
                key_test = create_key_MS([test_id], train=0, test=1)

                mask = np.array([True, True, True, True, False, False] * int(key_train_valid.shape[0]/6))
                np.random.shuffle(mask)
                key_train = key_train_valid[mask]
                key_valid = key_train_valid[~mask]

                X_train, Y_train = epoching(dic_data_train, key_train, steps_preprocess, DL=True)
                X_valid, Y_valid = epoching(dic_data_train, key_valid, steps_preprocess, DL=True)
                X_test, Y_test = epoching(dic_data_train, key_test, steps_preprocess, DL=True)

            elif within:
                print("Running a simple train/test set classification with each MS subject"
                      " trained and tested on their own data. Running subject {}".format(ID))   # for progress tracking
                cond = "W"
                key_train_valid = np.array([ID + "_1", ID + "_2"])  # grab the subject's training data
                key_test = np.array([ID + "_3", ID + "_4", ID + "_5", ID + "_6"])  # grab the subject's test data

                mask = np.array([True, False])  # only a length of 2 because we only have 2 training/validation files
                np.random.shuffle(mask)
                key_train = key_train_valid[mask]
                key_valid = key_train_valid[~mask]

                X_train, Y_train = epoching(dic_data_train, key_train, steps_preprocess, DL=True)
                X_valid, Y_valid = epoching(dic_data_train, key_valid, steps_preprocess, DL=True)
                X_test, Y_test = epoching(dic_data_test, key_test, steps_preprocess, DL=True)

            train_gen = DataGenerator(X_train, Y_train, 64)
            valid_gen = DataGenerator(X_valid, Y_valid, 64)
            test_gen = DataGenerator(X_test, Y_test, 64)

            pipelineDL = make_pipeline(LIMEd(DL=False),
                                       LIMEdl(channels=len(channels), valid_gen=valid_gen))

            # fit must get LIMEd data, since it will automatically un-lime it (valid_gen doesn't need this)
            train_gen.x = LIME_format(train_gen.x)
            test_gen.x = LIME_format(test_gen.x)

            pipelineDL.fit(train_gen.x, y=train_gen.y)
            score = pipelineDL.score(test_gen.x, y=test_gen.y)
            accuracy.append(
                dict(
                    subject=ID,
                    sess="NA",
                    classifier="DL",
                    condition=cond,
                    acc=score,
                    value=LIME_calc(train_gen.x, train_gen.y, test_gen.x, channels,
                                    pipelineDL.predict, sfreq)
                )
            )

            # now that we've completed one subject of MS DL, save results
            LIME_output(accuracy)
        elif SS:
            subset = {k: v for k, v in dic_data_train.items() if ID in k}  # subset to just the subject of interest
            sessions = ["S1", "S2", "S3"]
            for session in sessions:  # for each session
                if between:
                    print("Running a leave-one-out classification grouped by subject, leaving out {0} session {1}".format(subjects[subject], session))
                    cond = "B"
                    ID = subjects[subject] + "_" + session
                    train_key = {k: v for k, v in subset.items() if
                                 session not in k}  # train w/all sessions EXCEPT one (2)
                    test_key = {k: v for k, v in subset.items() if session in k}  # test on that session (1)

                    key_train_valid = np.array(list(train_key.keys()))  # convert to numpy array
                    key_test = np.array(list(test_key.keys()))

                    mask = np.array([True, True, True, True, True, True,
                                     True, True, True, True, True, True,
                                     True, True, True, True, True, True,
                                     False, False, False, False, False, False] * int(key_train_valid.shape[0] / 24))
                    np.random.shuffle(mask)
                    key_train = key_train_valid[mask]
                    key_valid = key_train_valid[~mask]

                    X_train, Y_train = epoching(dic_data_train, key_train, steps_preprocess, DL=True)
                    X_valid, Y_valid = epoching(dic_data_train, key_valid, steps_preprocess, DL=True)
                    X_test, Y_test = epoching(dic_data_train, key_test, steps_preprocess, DL=True)

                elif within:
                    print("Running a simple train/test set classification with each SS subject_session"
                          " trained and tested on their own data. Running subject {0} session {1}".format(subjects[subject], session))
                    cond = "W"
                    ID = subjects[subject] + "_" + session

                    key_train_valid = np.array([ID + "_1", ID + "_2", ID + "_3", ID + "_4"])  # grab the subject's training data
                    key_test = np.array([ID + "_5", ID + "_6", ID + "_7", ID + "_8",
                                         ID + "_9", ID + "_10", ID + "_11", ID + "_12"])  # grab the subject's test data

                    mask = np.array([True, True, True, False])
                    np.random.shuffle(mask)
                    key_train = key_train_valid[mask]
                    key_valid = key_train_valid[~mask]

                    X_train, Y_train = epoching(dic_data_train, key_train, steps_preprocess, DL=True)
                    X_valid, Y_valid = epoching(dic_data_train, key_valid, steps_preprocess, DL=True)
                    X_test, Y_test = epoching(dic_data_test, key_test, steps_preprocess, DL=True)

                train_gen = DataGenerator(X_train, Y_train, 64)
                valid_gen = DataGenerator(X_valid, Y_valid, 64)
                test_gen = DataGenerator(X_test, Y_test, 64)

                pipelineDL = make_pipeline(LIMEd(DL=False),
                                           LIMEdl(channels=len(channels), valid_gen=valid_gen))

                # fit must get LIMEd data, since it will automatically un-lime it (valid_gen doesn't need this)
                train_gen.x = LIME_format(train_gen.x)
                test_gen.x = LIME_format(test_gen.x)

                pipelineDL.fit(train_gen.x, y=train_gen.y)
                score = pipelineDL.score(test_gen.x, y=test_gen.y)
                accuracy.append(
                    dict(
                        subject=subjects[subject],
                        sess=session,
                        classifier="DL",
                        condition=cond,
                        acc=score,
                        value=LIME_calc(train_gen.x, train_gen.y, test_gen.x, channels,
                                        pipelineDL.predict, sfreq)
                    )
                )

            # now that we've completed one subject of SS DL (i.e. three sessions), save results
            LIME_output(accuracy)


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

            global sfreq
            sfreq = sub_data[0].info["sfreq"]  # save the sampling frequency for later
            global channels  # save the channel names for later
            channels = [sub_data[0].info["ch_names"][i] for i in mne.pick_types(sub_data[0].info, eeg=True)]
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
    print("Loading the SS dataset (all subjects and sessions that are available)...")  # for progress tracking
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

            global sfreq
            sfreq = sub_data[0].info["sfreq"]  # save the sampling frequency for later
            global channels  # save the channel names for later
            channels = [sub_data[0].info["ch_names"][i] for i in mne.pick_types(sub_data[0].info, eeg=True)]
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

