# XGBOD (Extreme Boosting Based Outlier Detection)
### Supplementary materials: datasets, demo source codes and sample outputs.

Y. Zhao and M.K. Hryniewicki, "XGBOD: Improving Supervised Outlier Detection with Unsupervised Representation Learning," *IJCNN '18*,  **Under Review**.

------------

Additional notes:
1. This repository is **temporary**. Would move to a **permanent** location later.
2. Codes are for **demo purpose only**. This demo codes are refactored for fast execution and reproduction as a proof of concept. The full codes will be released after the cleanup and optimization. In contrast to the demo version, the full version reserves the intermediate models to conduct feature engineering on testing data, which takes relatively long time to execute. However, the result difference is somehow neligible. However, it is noted users should not expose and use the testing data while building TOS. 

------------

##  Introduction
XGBOD is a three-phase framework (see Figure below). In the first phase, it generates new data representations. Specifically, various unsupervised outlier detection methods are applied to the original data to get transformed outlier scores as new data representations. In the second phase, a selection process is performed on newly generated outlier scores to keep the useful ones. The selected outlier scores are then combined with the original features to become the new feature space. Finally, an XGBoost model is trained on the new feature space, and its output decides the outlier prediction result.

![XGBOD Flowchart](https://github.com/yzhao062/XGBOD/blob/master/figs/flowchart.png "XGBOD Flowchart")

## Dependency
The experiement codes are writted in Python 3 and built on a number of Python packages:
- imbalanced_learn==0.3.2
- scipy==0.19.1
- numpy==1.13.1
- xgboost==0.7
- pandas==0.21.0
- PyNomaly==0.1.7
- imblearn==0.0
- scikit_learn==0.19.1

Batch installation is possible with the supplied "requirements.txt"
## Datasets
Five datasets are used (see dataset folder):

|  Datasets | Dimension  | Sample Size  | Number of Outliers  |
| ------------ | ------------ | ------------ | ------------ |
| Arrhythmia  | 351  | 274  | 126 (36%)  |
|  Letter | 1600  | 32  | 100 (6.25%)  |
|  Cardio | 1831  | 21  | 176 (9.6%)  |
|  Speech | 3686  | 600  | 61(1.65%)  |
|  Mammography | 11863  | 6  | 260 (2.32%)  |

All datasets are accesible from http://odds.cs.stonybrook.edu/. Citation Suggestion for the datasets please refer to: 
> Shebuti Rayana (2016).  ODDS Library [http://odds.cs.stonybrook.edu]. Stony Brook, NY: Stony Brook University, Department of Computer Science.

## Usage and Sample Output
Experiments could be reproduced by running **xgbod.py** directly. You could simply download/clone the entire repository and execute the code by "python xgbod.py".

The first part of the code read in the datasets using Scipy. Five public outlier datasets are supplied. Then various TOS are built by seven different algorithms:
1. KNN 
2. K-Median 
3. AvgKNN 
4. LOF
5. LoOP
6. One-Class SVM 
7. Isolation Forests
**Please be noted that you could include more TOS**

Taking KNN as an example, codes are as below:
```python
# Generate TOS using KNN based algorithms
feature_list, roc_knn, prc_n_knn, result_knn = get_TOS_knn(X_norm, y, k_range,
                                                                                               feature_list)
```
Then three TOS selection methods are used to select *p*  TOS:
```python

p = 10  # number of selected TOS
# random selection
X_train_new_rand, X_train_all_rand = random_select(X, X_train_new_orig,
                                                                                    roc_list, p)
# accurate selection
X_train_new_accu, X_train_all_accu = accurate_select(X, X_train_new_orig,
                                                                                     feature_list, roc_list, p)
# balance selection
X_train_new_bal, X_train_all_bal = balance_select(X, X_train_new_orig,
                                                                               roc_list, p)
```
Finally, various classification methods are applied to the datasets:

Sample outputs are provided below:
![Sample Outputs on Arrhythmia](https://github.com/yzhao062/XGBOD/blob/master/figs/sample_outputs.png "Sample Outputs on Arrhythmia")
![Sample Outputs on Arrhythmia2](https://github.com/yzhao062/XGBOD/blob/master/figs/results.png "Sample Outputs on Arrhythmia")

## Figures
To finish


