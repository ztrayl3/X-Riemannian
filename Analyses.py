from mne.decoding import CSP
from pyriemann.classification import MDM
from sklearn.pipeline import make_pipeline
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from INRIA import load_SS, load_MS, test_pipeline_DL, test_pipeline, LIMEd

# dictionary for all our Riemannian Geometry testing pipelines
pipelines = {}
pipelines['8csp+lda'] = make_pipeline(LIMEd(DL=False),
                                      CSP(n_components=8),
                                      LDA())  # baseline comparison CSP+LDA
pipelines['MDM'] = make_pipeline(LIMEd(DL=False),
                                 Covariances(estimator='lwf'),
                                 MDM(metric='riemann', n_jobs=-1))  # simple Riemannian
pipelines['tangentspace+LR'] = make_pipeline(LIMEd(DL=False),
                                             Covariances(estimator='lwf'),
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

    steps_preprocess = {"filter": [8, 30],  # filter from 8-30Hz
                        "drop_channels": ['EOG1', 'EOG2', 'EOG3', 'EMGg', 'EMGd'],  # ignore EOG/EMG channels
                        "tmin": 0.5, "tmax": 2.5, "overlap": 1, "length": 2}

    test_pipeline_DL(data, dic_data, steps_preprocess, MS=True, between=True)


#############################################################
# Analysis   : MS_DL_Within                                 #
# Dataset    : Multi Subject, Single Session                #
# Classifier : ShallowConvNet                               #
# Condition  : Within Subject Performance (train/test set)  #
#############################################################

def MS_DL_Within():
    data, dic_data_train, dic_data_test = load_MS(within=True)  # load multi-subject dataset for within subject analysis

    steps_preprocess = {"filter": [8, 30],  # filter from 8-30Hz
                        "drop_channels": ['EOG1', 'EOG2', 'EOG3', 'EMGg', 'EMGd'],  # ignore EOG/EMG channels
                        "tmin": 0.5, "tmax": 2.5, "overlap": 1, "length": 2}

    test_pipeline_DL(data, dic_data_train, steps_preprocess, dic_data_test=dic_data_test, MS=True, within=True)

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
                        "score": "LIME"}
    test_pipeline(dic_data, pipelines, session, steps_preprocess, between=True, MS=True)  # run it!


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
                        "score": "LIME"}
    test_pipeline(dic_data_test, pipelines, session, steps_preprocess, dic_data_train, within=True, MS=True)


########################################################################################################################


#############################################################
# Analysis   : SS_DL_Between                                #
# Dataset    : Single Subject, Multi Session                #
# Classifier : ShallowConvNet                               #
# Condition  : Between Session Performance (leave-one-out)  #
#############################################################


def SS_DL_Between():
    data, dic_data = load_SS(between=True)  # load multi-subject dataset for between subject analysis

    steps_preprocess = {"filter": [8, 30],  # filter from 8-30Hz
                        "drop_channels": ['EOG1', 'EOG2', 'EOG3', 'EMGg', 'EMGd'],  # ignore EOG/EMG channels
                        "tmin": 0.5, "tmax": 2.5, "overlap": 1, "length": 2}

    test_pipeline_DL(data, dic_data, steps_preprocess, SS=True, between=True)


#############################################################
# Analysis   : SS_DL_Within                                 #
# Dataset    : Single Subject, Multi Session                #
# Classifier : ShallowConvNet                               #
# Condition  : Within Session Performance (train/test set)  #
#############################################################


def SS_DL_Within():
    data, dic_data_train, dic_data_test = load_SS(within=True)  # load single-subject dataset for within subject analysis

    steps_preprocess = {"filter": [8, 30],  # filter from 8-30Hz
                        "drop_channels": ['EOG1', 'EOG2', 'EOG3', 'EMGg', 'EMGd'],  # ignore EOG/EMG channels
                        "tmin": 0.5, "tmax": 2.5, "overlap": 1, "length": 2}

    test_pipeline_DL(data, dic_data_train, steps_preprocess, dic_data_test=dic_data_test, SS=True, within=True)


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
                        "score": "LIME"}
    test_pipeline(dic_data, pipelines, target, steps_preprocess, between=True, SS=True)


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
                        "score": "LIME"}
    test_pipeline(dic_data_test, pipelines, target, steps_preprocess, dic_data_train, within=True, SS=True)
