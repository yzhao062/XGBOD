# XGBOD (Extreme Boosting Based Outlier Detection)
### Supplementary materials are provided, including datasets, source codes and sample outputs.
**Note: this is only a temporary site. May migrate to pemanent location later.**

##  Introduction
XGBOD is a two-phase approach that uses unsupervised outlier detection algorithms to improve the data representation and then applies an XGBoost classifier to predict on the improved feature space. Experiments on five popular outlier benchmark datasets show that XGBOD could achieve better results than various the state-of-the-art methods.

A high-level flowchart is supplied below:
![XGBOD Flowchart](https://github.com/yzhao062/XGBOD/blob/master/figs/flowchart.png "XGBOD Flowchart")

## Dependency
The experiement codes are writted in Python 3 and built on a number of Python packages:
- imbalanced_learn==0.3.2
- scipy==0.19.1
- numpy==1.13.1
- xgboost==0.6
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
Experiments could be reproduced by running **xgbod.py**.

The first part of the code read in the data using Scipy. Then various TOS are built by seven different algorithms:
1. KNN 
2. K-Median 
3. AvgKNN 
4. LOF
5. LoOP
6. One-Class SVM 
7. Isolation Forests

Taking KNN as an example, codes are as below:
```python
# # Generate TOS using KNN based algorithms
(feature_list, roc_knn, prec_knn, result_knn) = generate_TOS_knn(X_norm, y,
                                                                 k_list,
                                                                 feature_list)
```
Then three TOS selection methods are used to select *p* TOS:
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

A sample output is provided below:
![Sample Outputs on Arrhythmia](https://github.com/yzhao062/XGBOD/blob/master/figs/sample_outputs.png "Sample Outputs on Arrhythmia")

## Figures
To finish


