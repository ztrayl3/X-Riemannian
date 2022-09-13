import mne
import os

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
