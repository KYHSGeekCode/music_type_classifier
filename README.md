# music_type_classifier
CGSS music type classifier

- labels # : 4
- data #: about 400


## SVM
Accuracy 40%

## XGBoost
Accuracy 30%

## PCA
![img](https://github.com/KYHSGeekCode/music_type_classifier/blob/main/pca.png?raw=true)

## Pre-Process
Take 6,000 + features using [openSMILE built for M1](https://github.com/KYHSGeekCode/opensmile-python).
MinMax normalizer, MavVariance feature selection, etc.
