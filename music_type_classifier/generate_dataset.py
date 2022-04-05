import os

import numpy as np

from music_type_classifier.generate_label import get_labels
from music_type_classifier.settings import FEAT_FOLDER
from pandas import read_csv


def run():
    # target: [feature, label, music_id]
    labels = get_labels()  # label, music_id
    # read for each csvs in feature
    X = []
    y = []
    input_directory = os.fsencode(FEAT_FOLDER)
    processed = 0
    for file in os.listdir(input_directory):
        filename = os.fsdecode(file)
        music_id = int(filename.replace("feat_compare2016_", "").replace(".csv", ""))
        data = read_csv(os.path.join(FEAT_FOLDER, filename))
        del data['file']
        del data['start']
        del data['end']
        if music_id in labels:
            X.append(np.reshape(data[0:].to_numpy(), -1))
            y.append(labels[music_id])
        else:
            print(f"Could not find label for {music_id}")
        processed += 1
        print(processed, music_id)
    X = np.array(X, dtype=object)
    y = np.array(y)
    print(X.shape)
    print(y.shape)
    np.save("dataset_X.npy", X)
    np.save("dataset_y.npy", y)


if __name__ == '__main__':
    run()
