import pickle
import pandas as pd

# load LIME results
source = open("LIME_test.pkl", "rb")
LIME = pickle.load(source)
source.close()

# grab relevant information
subject = LIME["subject"]  # string
classifier = LIME["classifier"]  # string
LEFT = LIME["value"]["Left"]  # dictionary {string: float}
RIGHT = LIME["value"]["Right"]  # dictionary {string: float}

# create dataframe for storage
classes = ["left", "right"]
colnames = ["Predicted", "Channel", "Time", "Weight"]
df = []
for i in classes:
    for j in LEFT.keys():
        key = j
        if i == "left":
            value = LEFT[j]
        if i == "right":
            value = RIGHT[j]

        channel = key.split("-")[0][:-2]  # grab channel name, remove "_t" from end (string, 10-20 system)
        time = key.split("-")[1]  # grab timestamp (int, 0-511 or 0-255 depending on sampling rate)
        weight = value * 1e6  # grab values (scale for convenience)

        row = [i, channel, time, weight]
        df.append(row)

out = pd.DataFrame(df, columns=colnames)
filename = subject + "_" + classifier + ".csv"
out.to_csv(filename)
