# music_type_classifier
CGSS music type classifier

- labels # : 4
- data #: about 400


## SVM
Accuracy 44%

## XGBoost
Accuracy 30%

## Elbow scores
![img](https://github.com/KYHSGeekCode/music_type_classifier/blob/main/elbow.png?raw=true)

## 3-means Clustering
![img](https://github.com/KYHSGeekCode/music_type_classifier/blob/main/3cluster.png?raw=true)

## Just with labels
![img](https://github.com/KYHSGeekCode/music_type_classifier/blob/main/with_labels.png?raw=true)

## Parallel
![img](https://github.com/KYHSGeekCode/music_type_classifier/blob/main/parallelcoords.png?raw=true)


## PCA_97
![img](https://github.com/KYHSGeekCode/music_type_classifier/blob/main/pca_97.png?raw=true)

## PCA
![img](https://github.com/KYHSGeekCode/music_type_classifier/blob/main/pca.png?raw=true)

## Pre-Process
Take 6,000 + features using [openSMILE built for M1](https://github.com/KYHSGeekCode/opensmile-python).
MinMax normalizer, MavVariance feature selection, etc.
