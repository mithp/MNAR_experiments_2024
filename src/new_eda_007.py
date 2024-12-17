# -*- coding: utf-8 -*-
"""
Created on 27th August 2024
defintions updated, classes added to single file
@author: mithp
"""
import math
from sklearn.experimental import enable_iterative_imputer
import numpy as np
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesRegressor, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
from sklearn.model_selection import KFold
import platform
from MissingnessIntroducer_v3 import MissingnessIntroducer, ClassifierEvaluatorCV, ModifiedClassifierEvaluatorCV

# Define your data
n_samples = 1000
n_features = 50
bayes_rate = 0.50

# Define the number of times the experiment is repeated
n_repeats = 2

# Initialize an empty DataFrame to store results
results_df = pd.DataFrame(columns=['AUC','AUC_wo' ,'Accuracy','Accuracy_wo', 'Classifier_Type', 'Imputer', 'Missingness', 'Repeat', 'columns_imp_count'])


# Define imputers
imputers = {
    'complete_case': None,
    'mean': SimpleImputer(strategy='mean'),
    'median': SimpleImputer(strategy='median'),
    'knn': KNNImputer(n_neighbors=2),
    'mice': IterativeImputer(random_state=0),
    'tree': IterativeImputer(ExtraTreesRegressor(n_estimators=10, random_state=0), max_iter= 40)
}


for repeat in range(n_repeats):
    X = np.random.randn(n_samples, n_features)
    y = np.random.binomial(1, bayes_rate, n_samples)    

    # Choose random columns to introduce missingness into
    n_columns = np.random.randint(2,X.shape[1]+1)
    n_columns= math.floor((n_columns/2))
    #n_columns=1
    columns = np.random.choice(X.shape[1], size=n_columns, replace=False)
    
    #%

    # Create a list of all columns
    all_columns = np.arange(X.shape[1])
    # Find the complimentary columns
    columns_complimentary = np.setdiff1d(all_columns, columns)
    # Shuffle the complimentary columns
    np.random.shuffle(columns_complimentary)
    # Select the same number of complimentary columns as in 'columns'
    columns_complimentary = columns_complimentary[:n_columns]
    # Sort both columns and columns_complimentary
    columns = np.sort(columns)
    columns_complimentary = np.sort(columns_complimentary)

    #%
    
    
    threshold_mnar = np.median(X, axis=0)
    threshold_mar  = np.median(X[:, columns], axis=0)
    missingness_types = {
        'mnarY2': MissingnessIntroducer(X.copy(), y).introduce_mnar_y(prob=0.2),
        'mnarY4': MissingnessIntroducer(X.copy(), y).introduce_mnar_y(prob=0.4),
        'mnarY8': MissingnessIntroducer(X.copy(), y).introduce_mnar_y(prob=0.8),
        'mnarF2': MissingnessIntroducer(X.copy(), y).introduce_mnar_focused(threshold_mnar, prob=0.2),
        'mnarF4': MissingnessIntroducer(X.copy(), y).introduce_mnar_focused(threshold_mnar, prob=0.4),
        'mnarF8': MissingnessIntroducer(X.copy(), y).introduce_mnar_focused(threshold_mnar, prob=0.8),
        'mnarD2': MissingnessIntroducer(X.copy(), y).introduce_mnar_diffused(threshold_mnar, prob=0.2),
        'mnarD4': MissingnessIntroducer(X.copy(), y).introduce_mnar_diffused(threshold_mnar, prob=0.4),
        'mnarD8': MissingnessIntroducer(X.copy(), y).introduce_mnar_diffused(threshold_mnar, prob=0.8),
        'mar2':  MissingnessIntroducer(X.copy(), y ).introduce_mar(threshold_mar, prob=0.2, columns=columns, columns_complimentary=columns_complimentary),
        'mar4':  MissingnessIntroducer(X.copy(), y ).introduce_mar(threshold_mar, prob=0.4, columns=columns, columns_complimentary=columns_complimentary),
        'mar8':  MissingnessIntroducer(X.copy(), y ).introduce_mar(threshold_mar, prob=0.8, columns=columns, columns_complimentary=columns_complimentary),
        'mcar2': MissingnessIntroducer(X.copy(), y).introduce_mcar(prob=0.2),
        'mcar4': MissingnessIntroducer(X.copy(), y).introduce_mcar(prob=0.4),
        'mcar8': MissingnessIntroducer(X.copy(), y).introduce_mcar(prob=0.8)
    }
    
    for missingness_name, X_missing in missingness_types.items():
        for imputer_name, imputer in imputers.items():
            for linear_flag in range(4): #4
                if (imputer_name == 'complete_case') & (linear_flag != 0):# & (linear_flag != 3):
                    continue
                
                auc, accuracy, class_type = ClassifierEvaluatorCV(X_missing, y).evaluate_classifier_with_imputation(imputer, linear_flag,  n_splits=5)
                auc2, accuracy2, class_type2 = ModifiedClassifierEvaluatorCV(X_missing, y).evaluate_classifier_with_imputation(imputer, linear_flag, n_splits=5)
                new_row = pd.DataFrame({'AUC': [auc], 'AUC_wo': [auc2], 'Accuracy': [accuracy],'Accuracy_wo': [accuracy2], 'Classifier_Type': [class_type], 'Imputer': [imputer_name], 'Missingness': [missingness_name], 'Repeat': [repeat], 'columns_imp_count': [n_columns]})
                results_df = pd.concat([results_df, new_row], ignore_index=True)


#%%
from datetime import datetime
# Get current time and format it as a string
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')



if platform.system() == 'Windows':
    path_to_save= 'path/dfs_saved/'
    
else:
    path_to_save= '/path/dfs_to_store/'


filename = path_to_save+ f'missy_{timestamp}.csv'

# Save the DataFrame to a CSV file
results_df.to_csv(filename, index=False)









































