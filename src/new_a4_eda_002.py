 # -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 12:46:30 2024

A4 data in the pipeline
@author: mithp
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesRegressor, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import platform

if platform.system() == 'Windows':
    vmri = pd.read_csv("path/A4_VMRI_PRV2.csv") # Volumetric MRI
    curated_data = pd.read_csv("path/curated_a4.csv") # combined with R scripts from csv's
    path_to_save= 'D:/Work/a4_study/dfs_saved/'
    
else:
    vmri = pd.read_csv("path/A4_VMRI_PRV2.csv") # Volumetric MRI
    curated_data = pd.read_csv("path/curated_a4.csv") # combined with R scripts from csv's
    path_to_save= 'path/dfs_to_store/'


vrmi_cols= vmri.columns[2:][:-2].to_numpy() # vmri CSV is used only to get the feature names
a4_MRI_dataset=curated_data[vrmi_cols] #extract MRI dataset from curated since it has other modality which can used later.
a4_MRI_dataset['Hippocampal Occupancy']=curated_data['Hippocampal Occupancy']
a4_MRI_dataset['Amyloid eligibility']=curated_data['Amyloid eligibility']

apoe_dummy= pd.get_dummies(curated_data['APOE genotype'])
apoe_cols=apoe_dummy.columns

retired_dummy= pd.get_dummies(curated_data['Participant retired'])
retired_cols=retired_dummy.columns

marital_dummy= pd.get_dummies(curated_data['Marital status'])
marital_cols=marital_dummy.columns

gender_dummy= pd.get_dummies(curated_data['Sex'])
gender_cols=gender_dummy.columns

# Leveraging demogrpahics and coginitive scores for MRI imputation
all_X=curated_data[['BID',
                    'Amyloid eligibility',
                    'Age (yrs)',
                    'Education (yrs)',
                    'PACC','Digit symbol',
                    'FCSRT (2xFree + Cued)',
                    'Logical memory delay',
                    'MMSE']]

all_X[apoe_cols]=apoe_dummy#Adding APOE to the mix
all_X[retired_cols]=retired_dummy#Adding retirement info to the mix
all_X[marital_cols]=marital_dummy#Adding marital to the mix
all_X[gender_cols]=gender_dummy#Adding gender to the mix

all_plus_mri_cols=np.concatenate((vrmi_cols, all_X.columns))
curated_for_imputation=curated_data[a4_MRI_dataset.columns]
curated_for_imputation[all_X.columns]=all_X

curated_for_imputation['MRI absent']=curated_for_imputation['LeftLateralVentricle'].isnull()# True means absent
#%%

clf=HistGradientBoostingClassifier()
jojo=curated_for_imputation.columns

X=curated_for_imputation.drop(columns={'MRI absent', 'Amyloid eligibility','BID'}).reset_index(drop=True)
# enc = OneHotEncoder(handle_unknown='ignore')
y=curated_for_imputation['Amyloid eligibility'].reset_index(drop=True)
y_one_hot=pd.get_dummies(y, drop_first=True)

#%%
from MissingnessIntroducer_v3 import ClassifierEvaluatorCV_scale_balance, ModifiedClassifierEvaluatorCV_scale_balance
# Define your data
# n_samples = 1000
# n_features = 50
# bayes_rate = 0.50

# # Define the probabilities for MCAR missingness
# #probs = [0, 0.2, 0.4, 0.8]
# probs = [0.2]

# Define the number of times the experiment is repeated
n_repeats = 20

# Initialize an empty DataFrame to store results
results_df = pd.DataFrame(columns=['AUC','AUC_wo' ,'Accuracy','Accuracy_wo', 'Classifier_Type', 'Imputer', 'Repeat'])


# Define imputers
imputers = {
    'mean': SimpleImputer(strategy='mean'),
    'median': SimpleImputer(strategy='median'),
    'knn': KNNImputer(n_neighbors=2),
    'mice': IterativeImputer(random_state=0),
    'tree': IterativeImputer(ExtraTreesRegressor(n_estimators=10, random_state=0))
}


for repeat in range(n_repeats):
    for imputer_name, imputer in imputers.items():
            for linear_flag in range(4): #4
                if (imputer_name == 'complete_case') & (linear_flag != 0):# & (linear_flag != 3):
                    continue
                X_missing= X.copy().to_numpy()
                y= y_one_hot.copy().to_numpy()
                auc, accuracy, class_type = ClassifierEvaluatorCV_scale_balance(X_missing, y).evaluate_classifier_with_imputation(imputer, linear_flag,  n_splits=5)
                auc2, accuracy2, class_type2 = ModifiedClassifierEvaluatorCV_scale_balance(X_missing, y).evaluate_classifier_with_imputation(imputer, linear_flag, n_splits=5)
                new_row = pd.DataFrame({'AUC': [auc], 'AUC_wo': [auc2], 'Accuracy': [accuracy],'Accuracy_wo': [accuracy2], 'Classifier_Type': [class_type], 'Imputer': [imputer_name], 'Repeat': [repeat]})
                results_df = pd.concat([results_df, new_row], ignore_index=True)


#%%
from datetime import datetime
# Get current time and format it as a string
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Append the timestamp to the filename
#path_to_save= 'path/dfs_to_store/'
filename = path_to_save+ f'data_a4_study_{timestamp}.csv'

# Save the DataFrame to a CSV file
results_df.to_csv(filename, index=False)


#%%
# Figure plot

# path_to_load ='path\dfs_to_store\data_a4_study_20240912_052102.csv'
# from datetime import datetime
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# df_to_plt= pd.read_csv(path_to_load)

# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# path_to_save_fig= 'path/figs_saved/'
# filename_fig_svg = path_to_save_fig+'with_wo_comp_'+ f'data_a4_{timestamp}.svg'
# filename_fig_png = path_to_save_fig+'with_wo_comp_'+ f'data_a4_{timestamp}.png'


# #%
# plt.figure(figsize=(8, 18))

# # Create a boxplot for AUC
# plt.subplot(2, 1, 1)
# #sns.boxplot(x='Imputer', y='AUC', hue='Classifier_Type', data=df_to_plt)
# sns.boxplot(x='Classifier_Type', y='AUC', hue='Imputer', data=df_to_plt)
# plt.title('AUC with Train Test Split')
# plt.legend(loc='upper right')
# plt.axhline(0.72, color='blue')  # Add a horizontal blue line at AUC 0.5
# plt.grid(True)  # Add a grid

# # Create a boxplot for AUC_wo
# plt.subplot(2, 1, 2)
# #sns.boxplot(x='Imputer', y='AUC_wo', hue='Classifier_Type', data=df_to_plt)
# sns.boxplot(x='Classifier_Type', y='AUC_wo', hue='Imputer', data=df_to_plt)
# plt.title('AUC without Train Test Split')
# plt.legend(loc='upper right')
# plt.axhline(0.72, color='blue')  # Add a horizontal blue line at AUC 0.5
# plt.grid(True)  # Add a grid

# plt.tight_layout()
# plt.title('Data A4 with and without data split')
# plt.savefig(filename_fig_svg, format='svg')
# plt.savefig(filename_fig_png, format='png')
# plt.show()















