import numpy as np
from INRIA import create_key_MS, create_key_SS, epoching, DataGenerator, load_MS, load_SS, test_pipeline
from EEGModels import ShallowConvNet, square, log
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import load_model
import torch
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import Covariances
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

# Deep learning specific parameters
input_window_samples = 1024
n_epochs = 500
cuda = torch.cuda.is_available()
n_classes = 2
batch_size = 32

# dictionary for all our testing pipelines
pipelines = {}
#pipelines['8csp+lda'] = make_pipeline(CSP(n_components=8), LDA())  # baseline comparison CSP+LDA
pipelines['MDM'] = make_pipeline(Covariances(estimator='lwf'), MDM(metric='riemann', n_jobs=-1))  # simple Riemannian
pipelines['tangentspace+LR'] = make_pipeline(Covariances(estimator='lwf'),
                                             TangentSpace(metric='riemann'),
                                             LogisticRegression(max_iter=350, n_jobs=-1))  # more realistic Riemannian

# session IDs for the single subject dataset
sessions = ["S1", "S2", "S3"]

#############################################################
# Analysis   : MS_DL_Between                                #
# Dataset    : Multi Subject, Single Session                #
# Classifier : ShallowConvNet                               #
# Condition  : Between Subject Performance (leave-one-out)  #
#############################################################


def MS_DL_Between():
    data, dic_data = load_MS(between=True)  # load multi-subject dataset for between subject analysis
    n_chans = data[list(data)[0]][0].get_channel_types().count("eeg")  # load the first file, count how many eeg channels

    steps_preprocess = {"filter": [8, 30],  # filter from 8-30Hz
                        "drop_channels": ['EOG1', 'EOG2', 'EOG3', 'EMGg', 'EMGd'],  # ignore EOG/EMG channels
                        "tmin": 0.5, "tmax": 2.5, "overlap": 1, "length": 2}

    my_callbacks = [
        EarlyStopping(patience=50, monitor="loss"),
        ModelCheckpoint(filepath='Model/best_model.h5', save_best_only=True)
    ]

    performance = {}
    subjects = list(data.keys())  # a list of participants to be used for analysis

    for subject in (range(len(subjects))):  # for each subject

        train_id = subjects.copy()  # train on all
        test_id = train_id.pop(subject)  # except for one (leave one out)

        key_train_valid = create_key_MS(train_id, train=1, test=1)
        key_test = create_key_MS([test_id], train=0, test=1)

        mask = np.array([True, True, True, True, False, False] * int(key_train_valid.shape[0]/6))
        np.random.shuffle(mask)
        key_train = key_train_valid[mask]
        key_valid = key_train_valid[~mask]

        X_train, Y_train = epoching(dic_data, key_train, steps_preprocess, DL=True)
        X_valid, Y_valid = epoching(dic_data, key_valid, steps_preprocess, DL=True)
        X_test, Y_test = epoching(dic_data, key_test, steps_preprocess, DL=True)

        train_gen = DataGenerator(X_train, Y_train, 64)
        valid_gen = DataGenerator(X_valid, Y_valid, 64)
        test_gen = DataGenerator(X_test, Y_test, 64)

        model = ShallowConvNet(nb_classes=n_classes, Chans=n_chans, Samples=input_window_samples)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=BinaryAccuracy())

        fit_model = model.fit(x=train_gen,
                              batch_size=batch_size,
                              epochs=n_epochs,
                              callbacks=my_callbacks,
                              validation_data=valid_gen,
                              verbose=1)

        model = load_model("Model/best_model.h5", custom_objects={"square": square, "log": log})

        performance[subject] = model.evaluate(x=test_gen)[1]
        # format: performance[subject] = accuracy if subject left out

    return performance


#############################################################
# Analysis   : MS_DL_Within                                 #
# Dataset    : Multi Subject, Single Session                #
# Classifier : ShallowConvNet                               #
# Condition  : Within Subject Performance (train/test set)  #
#############################################################

def MS_DL_Within():
    data, dic_data_train, dic_data_test = load_MS(within=True)  # load multi-subject dataset for within subject analysis
    n_chans = data[list(data)[0]][0].get_channel_types().count("eeg")  # load the first file, count how many eeg channels

    steps_preprocess = {"filter": [8, 30],  # filter from 8-30Hz
                        "drop_channels": ['EOG1', 'EOG2', 'EOG3', 'EMGg', 'EMGd'],  # ignore EOG/EMG channels
                        "tmin": 0.5, "tmax": 2.5, "overlap": 1, "length": 2}

    my_callbacks = [
        EarlyStopping(patience=50, monitor="loss"),
        ModelCheckpoint(filepath='Model/best_model.h5', save_best_only=True)
    ]

    performance = {}
    subjects = list(data.keys())  # a list of participants to be used for analysis

    for subject in (range(len(subjects))):  # for each subject
        ID = subjects[subject]  # training and testing on the same subject!

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

        model = ShallowConvNet(nb_classes=n_classes, Chans=n_chans, Samples=input_window_samples)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=BinaryAccuracy())

        fit_model = model.fit(x=train_gen,
                              batch_size=batch_size,
                              epochs=n_epochs,
                              callbacks=my_callbacks,
                              validation_data=valid_gen,
                              verbose=1)

        model = load_model("Model/best_model.h5", custom_objects={"square": square, "log": log})

        performance[ID] = model.evaluate(x=test_gen)[1]
        # format: performance[subject] = accuracy of within-subject classification

    return performance


#############################################################
# Analysis   : MS_RG_Between                                #
# Dataset    : Multi Subject, Single Session                #
# Classifier : Riemannian Geometry (and others)             #
# Condition  : Between Subject Performance (leave-one out)  #
#############################################################


def MS_RG_Between():
    # load the MS dataset
    data, dic_data = load_MS(between=True)

    # run the selected pipelines
    session = list(data.keys())  # a list of participants to be used for analysis
    steps_preprocess = {"filter": [8, 30],  # filter from 8-30Hz
                        "drop_channels": ['EOG1', 'EOG2', 'EOG3', 'EMGg', 'EMGd'],  # ignore EOG/EMG channels
                        "tmin": 0.5, "tmax": 4, "overlap": 1/16, "length": 1,
                        "score": "EAcc"}
    accuracy = test_pipeline(dic_data, pipelines, session, steps_preprocess,
                             between=True, MS=True)  # run it!

    return accuracy


#############################################################
# Analysis   : MS_RG_Within                                 #
# Dataset    : Multi Subject, Single Session                #
# Classifier : Riemannian Geometry (and others)             #
# Condition  : Within Subject Performance (train/test set)  #
#############################################################


def MS_RG_Within():
    # load the MS dataset
    data, dic_data_train, dic_data_test = load_MS(within=True)

    # run the selected pipelines
    session = list(data.keys())  # a list of participants to be used for analysis
    steps_preprocess = {"filter": [8, 30],  # filter from 8-30Hz
                        "drop_channels": ['EOG1', 'EOG2', 'EOG3', 'EMGg', 'EMGd'],  # ignore EOG/EMG channels
                        "tmin": 0.5, "tmax": 4, "overlap": 1/16, "length": 1,
                        "score": "EAcc"}
    accuracy = test_pipeline(dic_data_test, pipelines, session, steps_preprocess, dic_data_train,
                             within=True, MS=True)

    return accuracy


########################################################################################################################


#############################################################
# Analysis   : SS_DL_Between                                #
# Dataset    : Single Subject, Multi Session                #
# Classifier : ShallowConvNet                               #
# Condition  : Between Session Performance (leave-one-out)  #
#############################################################


#############################################################
# Analysis   : SS_DL_Within                                 #
# Dataset    : Single Subject, Multi Session                #
# Classifier : ShallowConvNet                               #
# Condition  : Within Session Performance (train/test set)  #
#############################################################


#############################################################
# Analysis   : SS_RG_Between                                #
# Dataset    : Single Subject, Multi Session                #
# Classifier : Riemannian Geometry (and others)             #
# Condition  : Between Session Performance (leave-one out)  #
#############################################################


def SS_RG_Between():
    # load the SS dataset
    data, dic_data = load_SS(between=True)
    subjects = list(data.keys())  # a list of participants to be used for analysis
    target = []
    for sub in subjects:
        for sess in sessions:
            target.append(sub + "_" + sess)  # create a joint list of subject-session IDs to fit the pipeline format

    steps_preprocess = {"filter": [8, 30],  # filter from 8-30Hz
                        "drop_channels": ['EOG1', 'EOG2', 'EOG3', 'EMGg', 'EMGd'],  # ignore EOG/EMG channels
                        "tmin": 0.5, "tmax": 4, "overlap": 1 / 16, "length": 1,
                        "score": "EAcc"}
    accuracy = test_pipeline(dic_data, pipelines, target, steps_preprocess,
                             between=True, SS=True)

    return accuracy


#############################################################
# Analysis   : SS_RG_Within                                 #
# Dataset    : Single Subject, Multi Session                #
# Classifier : Riemannian Geometry (and others)             #
# Condition  : Within Session Performance (train/test set)  #
#############################################################


def SS_RG_Within():
    # load the SS dataset
    data, dic_data_train, dic_data_test = load_SS(within=True)
    subjects = list(data.keys())  # a list of participants to be used for analysis
    target = []
    for sub in subjects:
        for sess in sessions:
            target.append(sub + "_" + sess)   # create a joint list of subject-session IDs to fit the pipeline format

    steps_preprocess = {"filter": [8, 30],  # filter from 8-30Hz
                        "drop_channels": ['EOG1', 'EOG2', 'EOG3', 'EMGg', 'EMGd'],  # ignore EOG/EMG channels
                        "tmin": 0.5, "tmax": 4, "overlap": 1 / 16, "length": 1,
                        "score": "EAcc"}
    accuracy = test_pipeline(dic_data_test, pipelines, target, steps_preprocess, dic_data_train,
                             within=True, SS=True)

    return accuracy
