import mne
import os

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
        data[sub] = sub_data  # save sub_data list into data dictionary

# Current data format: data[subject] holds all 30 raw objects
# data[subject][0] = Closed eyes baseline
# data[subject][1] = Open eyes baseline
# data[subject][2] = Nback task 1
# data[subject][3] = Nback task 2
# data[subject][4] = Nback task 3
# data[subject][5] = Nback task 4
# data[subject][6] = Nback task 5
# data[subject][7] = Nback task 6
# data[subject][8] = Training session 1
# data[subject][9] = Training session 2
# data[subject][10] = Training session 3
# data[subject][11] = Training session 4
# data[subject][12] = Online session 1 (w/ feedback)
# data[subject][13] = Online session 2 (w/ feedback)
# data[subject][14] = Online session 3 (w/ feedback)
# data[subject][15] = Online session 4 (w/ feedback)
# data[subject][16] = Online session 5 (w/ feedback)
# data[subject][17] = Online session 6 (w/ feedback)
# data[subject][18] = Online session 7 (w/ feedback)
# data[subject][19] = Online session 8 (w/ feedback)
# data[subject][20] = Reading span task 1
# data[subject][21] = Reading span task 2
# data[subject][22] = Reading span task 3
# data[subject][23] = Reading span task 4
# data[subject][24] = Reading span task 5
# data[subject][25] = Reading span task 6
# data[subject][26] = Reading span task 7
# data[subject][27] = Reading span task 8
# data[subject][28] = Reading span task 9
# data[subject][29] = Reading span task 10
