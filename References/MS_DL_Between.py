import mne
import os
import numpy as np

from EEGModels import ShallowConvNet, square, log
from tensorflow.keras.utils import Sequence
from tensorflow import one_hot
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import load_model
import torch

from numpy.random import seed
seed(2002012)


########################################
# DEFINE FUNCTIONS FROM INRIA PIPELINE #
########################################

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


def epoching(dict, key_session=[], steps_preprocess=None, key_events={"769":0 ,"770":1}):
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


#################
# LOAD THE DATA #
#################

# Establish master data path
path = os.path.join("../Data", "MS")

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
for i in data.keys():  # for every subject...
    for j in range(1, 7):
        session = str(i) + '_' + str(j)  # their numbering will start from 1...
        dic_data[session] = data[i][j + 1]  # but indexing must follow the indexes from the comment above (hence +1)


#########################
# RUN INRIA DL PIPELINE #
#########################

n_chans = 27
input_window_samples = 1024
# sfreq = dic_data_train['A1_1'].info["sfreq"]
# ch_names = dic_data_train['A1_1'].info["ch_names"]
n_epochs = 500
cuda = torch.cuda.is_available()
# device = 'cuda' if cuda else 'cpu'
n_classes = 2
batch_size = 32

steps_preprocess = {"filter": [8, 30],  # filter from 8-30Hz
                    "drop_channels": ['EOG1', 'EOG2', 'EOG3', 'EMGg', 'EMGd'],  # ignore EOG/EMG channels
                    "tmin": 0.5, "tmax": 2.5, "overlap": 1, "length": 2}

my_callbacks = [
    EarlyStopping(patience=50, monitor="loss"),
    ModelCheckpoint(filepath='../Model/best_model.h5', save_best_only=True)
]

performance = {}
subjects = list(data.keys())  # a list of participants to be used for analysis

for subject in (range(len(subjects))):  # for each subject

    train_id = subjects.copy()  # train on all
    test_id = train_id.pop(subject)  # except for one (leave one out)

    key_train_valid = create_key(train_id, train=1, test=1)
    key_test = create_key([test_id], train=0, test=1)

    mask = np.array([True, True, True, True, False, False] * int(key_train_valid.shape[0]/6))
    np.random.shuffle(mask)
    key_train = key_train_valid[mask]
    key_valid = key_train_valid[~mask]

    X_train, Y_train = epoching(dic_data, key_train, steps_preprocess)
    X_valid, Y_valid = epoching(dic_data, key_valid, steps_preprocess)
    X_test, Y_test = epoching(dic_data, key_test, steps_preprocess)

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

    model = load_model("../Model/best_model.h5", custom_objects={"square": square, "log": log})

    performance[subject] = model.evaluate(x=test_gen)[1]  # format: performance[subject] = accuracy if subject left out
