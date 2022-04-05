# generate label (music_id -> type) from db.
import os

import opensmile

from music_type_classifier.settings import FEAT_FOLDER, MUSIC_FOLDER


def preprocess_wavs(input_path: str, out_path: str):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    input_directory = os.fsencode(input_path)
    os.makedirs(out_path, exist_ok=True)
    for file in os.listdir(input_directory):
        filename = os.fsdecode(file)
        music_id = int(filename.replace("song_", "").replace(".wav", ""))
        y = smile.process_file(os.path.join(input_path, filename))
        y.to_csv(os.path.join(out_path, f"feat_compare2016_{music_id}.csv"))
        print(filename)


if __name__ == '__main__':
    preprocess_wavs(MUSIC_FOLDER, FEAT_FOLDER)
