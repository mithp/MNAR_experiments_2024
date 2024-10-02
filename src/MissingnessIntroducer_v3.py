# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 13:54:43 2024

@author: mithp
"""

import numpy as np
import math

import math
from sklearn.experimental import enable_iterative_imputer
import numpy as np
#from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesRegressor, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

class MissingnessIntroducer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        #self.columns = columns if columns is not None else []

    def introduce_missingness(self, X, mask):
        X[mask] = np.nan
        return X
    
    def introduce_mnar_diffused(self, thresholds, prob):
        for row in range(self.X.shape[0]):
            mask_y_value = (self.y[row] == 1)
            if mask_y_value:
                mask_threshold = self.X[row, :] > thresholds
                mask_intersection = mask_threshold & mask_y_value
                masked_indices = np.where(mask_intersection)[0]
                num_to_select = int(prob * len(masked_indices))
                if num_to_select > 0:
                    selected_indices = np.random.choice(masked_indices, size=num_to_select, replace=False)
                    self.X[row, selected_indices] = self.introduce_missingness(self.X[row, selected_indices], True)
        return self.X
    
    def introduce_mnar_y_full(self, prob):
        #this selectls all y row indices
        X_array = self.X
        row_indices = np.where(self.y == 1)[0]
        
        # Create a mask for the entire array
        mask = np.zeros_like(X_array, dtype=bool)
        
        for row in row_indices:
            num_elements = X_array.shape[1]
            num_selected = int(num_elements * prob)
            selected_indices = np.random.choice(num_elements, num_selected, replace=False)
            mask[row, selected_indices] = True
        
        # Apply the mask to the array
        self.X = self.introduce_missingness(self.X, mask)
    
        return self.X
    
    def introduce_mnar_y(self, prob):
        # this selects only subset of y indices
        X_array = self.X
        row_indices = np.where(self.y == 1)[0]
        num_y_to_select = int(prob * len(row_indices))
        row_indices_to_select = np.random.choice(row_indices, size=num_y_to_select, replace=False)
        
        # Create a mask for the entire array
        mask = np.zeros_like(X_array, dtype=bool)
        
        for row in row_indices_to_select:
            num_elements = X_array.shape[1]
            num_selected = int(num_elements * prob)
            selected_indices = np.random.choice(num_elements, num_selected, replace=False)
            mask[row, selected_indices] = True
        
        # Apply the mask to the array
        self.X = self.introduce_missingness(self.X, mask)
    
        return self.X
        
    def introduce_mnar_focused(self, thresholds, prob):
        mask_threshold = self.X > thresholds
        masked_indices = np.where(mask_threshold)
        
        for row in np.unique(masked_indices[0]):
            row_indices = masked_indices[1][masked_indices[0] == row]
            num_to_select = int(prob * len(row_indices))
            
            if num_to_select > 0:
                selected_indices = np.random.choice(row_indices, size=num_to_select, replace=False)
                self.X[row, selected_indices] = self.introduce_missingness(self.X[row, selected_indices], True)
        
        return self.X
        
    def introduce_mar(self,thresholds, prob, columns, columns_complimentary):
        for column, threshold, column_comp in zip(columns, thresholds, columns_complimentary):
            mask_threshold = (self.X[:, column] > threshold)
            masked_indices = np.where(mask_threshold)[0]
            # Apply probability
            mask_indices_prob_selected = np.random.choice(masked_indices, size=int(prob * len(masked_indices)), replace=False)
            mask = np.zeros_like(self.X[:, column_comp] , dtype=bool)
            mask[mask_indices_prob_selected] = True
            self.X[:, column_comp] = self.introduce_missingness(self.X[:, column_comp], mask)
        return self.X
    
    def introduce_mcar(self, prob):
        # Create a mask for the entire dataset based on the given probability
        mask = np.random.choice([True, False], size=self.X.shape, p=[prob, 1-prob])
        
        # Introduce missingness based on the mask
        self.X[mask] = self.introduce_missingness(self.X[mask], True)
        
        return self.X

class ClassifierEvaluatorCV:
  def __init__(self, X, y):
    self.X = X
    self.y = y

  def evaluate_classifier_with_imputation(self, imputer=None, linear_flag=0, n_splits=5):
    """
    Evaluates classifiers using k-fold cross-validation and stores mean AUC.

    Args:
      imputer: Imputer object for handling missing values (optional).
      linear_flag: Flag to choose classifier (0: GBM, 1: LDA, 2: SVC, 3: RF).
      n_splits: Number of folds for k-fold cross-validation (default: 5).

    Returns:
      mean_auc: The average Area Under the ROC Curve (AUC) across folds.
      accuracies: A list of accuracy scores for each fold.
      class_type: String representing the evaluated classifier type.
    """

    classifiers = {
      0: (HistGradientBoostingClassifier(), 'GB'),
      1: (LinearDiscriminantAnalysis(), 'LDA'),
      2: (CalibratedClassifierCV(SVC(C=0.1, kernel='linear', probability=True, random_state=1)), 'SVC'),
      #2: (SVC(C=0.1, kernel='linear',  probability=True,random_state=1), 'SVC'),
      3: (RandomForestClassifier(max_depth=2, random_state=0), 'RF')
      #4: (QuadraticDiscriminantAnalysis(), 'QDA')
    }

    clf, class_type = classifiers[linear_flag]
    #kfold = KFold(n_splits=n_splits, shuffle=True)
    # Initialize StratifiedKFold
    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    aucs, accuracies = [], []

    for train_index, test_index in stratified_kfold.split(self.X, self.y):
      X_train, X_test = self.X[train_index], self.X[test_index]
      y_train, y_test = self.y[train_index], self.y[test_index]

      if (imputer is not None) & (linear_flag != 0):
          X_train = imputer.fit_transform(X_train)
          X_test = imputer.transform(X_test)
      clf.fit(X_train, y_train)
      y_prob = clf.predict_proba(X_test)[:, 1]
      auc = roc_auc_score(y_test, y_prob)
      accuracy = accuracy_score(y_test, np.round(y_prob))

      aucs.append(auc)
      accuracies.append(accuracy)

    mean_auc = np.mean(aucs)
    mean_acc = np.mean(accuracies)
    return mean_auc, mean_acc, class_type

class ModifiedClassifierEvaluatorCV:
  def __init__(self, X, y):
    self.X = X
    self.y = y

  def evaluate_classifier_with_imputation(self, imputer=None, linear_flag=0, n_splits=5):
    """
    Evaluates classifiers with imputation (before train/test split) and k-fold CV.

    Args:
      imputer: Imputer object for handling missing values (optional).
      linear_flag: Flag to choose classifier (0: GBM, 1: LDA, 2: SVC, 3: RF).
      n_splits: Number of folds for k-fold cross-validation (default: 5).

    Returns:
      mean_auc: The average Area Under the ROC Curve (AUC) across folds.
      accuracies: A list of accuracy scores for each fold.
      class_type: String representing the evaluated classifier type.
    """

    classifiers = {
      0: (HistGradientBoostingClassifier(), 'GB'),
      1: (LinearDiscriminantAnalysis(), 'LDA'),
      2: (CalibratedClassifierCV(SVC(C=0.1, kernel='linear', probability=True, random_state=1)), 'SVC'),
      #2: (SVC(C=0.1,kernel='linear',  probability=True,random_state=1), 'SVC'),
      3: (RandomForestClassifier(max_depth=2, random_state=0), 'RF')
      #4: (QuadraticDiscriminantAnalysis(), 'QDA')
    }

    clf, class_type = classifiers[linear_flag]
    #kfold = KFold(n_splits=n_splits, shuffle=True)
    # Initialize StratifiedKFold
    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    aucs, accuracies = [], []

    # Impute on the entire dataset before CV
    if imputer is not None:
      self.X = imputer.fit_transform(self.X)

    for train_index, test_index in stratified_kfold.split(self.X, self.y):
      X_train, X_test = self.X[train_index], self.X[test_index]
      y_train, y_test = self.y[train_index], self.y[test_index]
      clf.fit(X_train, y_train)
      y_prob = clf.predict_proba(X_test)[:, 1]
      auc = roc_auc_score(y_test, y_prob)
      accuracy = accuracy_score(y_test, np.round(y_prob))

      aucs.append(auc)
      accuracies.append(accuracy)

    mean_auc = np.mean(aucs)
    mean_acc = np.mean(accuracies)
    return mean_auc, mean_acc, class_type

#%% ADNI n PPMI

class ClassifierEvaluatorCV_scale:
  def __init__(self, X, y):
    self.X = X
    self.y = y

  def evaluate_classifier_with_imputation(self, imputer=None, linear_flag=0, n_splits=5):
    """
    Evaluates classifiers using k-fold cross-validation and stores mean AUC.

    Args:
      imputer: Imputer object for handling missing values (optional).
      linear_flag: Flag to choose classifier (0: GBM, 1: LDA, 2: SVC, 3: RF).
      n_splits: Number of folds for k-fold cross-validation (default: 5).

    Returns:
      mean_auc: The average Area Under the ROC Curve (AUC) across folds.
      accuracies: A list of accuracy scores for each fold.
      class_type: String representing the evaluated classifier type.
    """

    classifiers = {
      0: (HistGradientBoostingClassifier(), 'GB'),
      1: (LinearDiscriminantAnalysis(), 'LDA'),
      2: (CalibratedClassifierCV(SVC(C=0.1, kernel='linear', probability=True, random_state=1)), 'SVC'),
      3: (RandomForestClassifier(max_depth=2, random_state=0), 'RF')
    }

    clf, class_type = classifiers[linear_flag]
    kfold = KFold(n_splits=n_splits, shuffle=True)
    aucs, accuracies = [], []

    for train_index, test_index in kfold.split(self.X):
      X_train, X_test = self.X[train_index], self.X[test_index]
      y_train, y_test = self.y[train_index], self.y[test_index]

      if (imputer is not None) & (linear_flag != 0):
          X_train_raw = imputer.fit_transform(X_train)
          X_test_raw = imputer.transform(X_test)
          scaler = StandardScaler()
          X_train= scaler.fit_transform(X_train_raw)
          X_test= scaler.transform(X_test_raw)

      if X_train.shape[0] < 6 or X_test.shape[0] < 6 or X_train.shape[1] < 4 or X_test.shape[1] < 4:
        auc = np.nan
        accuracy = np.nan
      else:
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]

        try:
          auc = roc_auc_score(y_test, y_prob)
        except ValueError as e:
          print(e)
          auc = None  # or choose a specific value

        accuracy = accuracy_score(y_test, np.round(y_prob))

      aucs.append(auc)
      accuracies.append(accuracy)

    mean_auc = np.mean(aucs)
    mean_acc = np.mean(accuracies)
    return mean_auc, mean_acc, class_type

class ModifiedClassifierEvaluatorCV_scale:
  def __init__(self, X, y):
    self.X = X
    self.y = y

  def evaluate_classifier_with_imputation(self, imputer=None, linear_flag=0, n_splits=5):
    """
    Evaluates classifiers with imputation (before train/test split) and k-fold CV.

    Args:
      imputer: Imputer object for handling missing values (optional).
      linear_flag: Flag to choose classifier (0: GBM, 1: LDA, 2: SVC, 3: RF).
      n_splits: Number of folds for k-fold cross-validation (default: 5).

    Returns:
      mean_auc: The average Area Under the ROC Curve (AUC) across folds.
      accuracies: A list of accuracy scores for each fold.
      class_type: String representing the evaluated classifier type.
    """

    classifiers = {
      0: (HistGradientBoostingClassifier(), 'GB'),
      1: (LinearDiscriminantAnalysis(), 'LDA'),
      2: (CalibratedClassifierCV(SVC(C=0.1, kernel='linear', probability=True, random_state=1)), 'SVC'),
      3: (RandomForestClassifier(max_depth=2, random_state=0), 'RF')
    }

    clf, class_type = classifiers[linear_flag]
    kfold = KFold(n_splits=n_splits, shuffle=True)
    aucs, accuracies = [], []

    # Impute on the entire dataset before CV
    if imputer is not None:
      self.X = imputer.fit_transform(self.X)

    for train_index, test_index in kfold.split(self.X):
      X_train_raw, X_test_raw = self.X[train_index], self.X[test_index]
      y_train, y_test = self.y[train_index], self.y[test_index]
      scaler = StandardScaler()
      X_train= scaler.fit_transform(X_train_raw)
      X_test= scaler.transform(X_test_raw)

      if X_train.shape[0] < 6 or X_test.shape[0] < 6 or X_train.shape[1] < 4 or X_test.shape[1] < 4:
        auc = np.nan
        accuracy = np.nan
      else:
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]

        try:
          auc = roc_auc_score(y_test, y_prob)
        except ValueError as e:
          print(e)
          auc = None  # or choose a specific value

        accuracy = accuracy_score(y_test, np.round(y_prob))

      aucs.append(auc)
      accuracies.append(accuracy)

    mean_auc = np.mean(aucs)
    mean_acc = np.mean(accuracies)
    return mean_auc, mean_acc, class_type

#%% stratified 


class ClassifierEvaluatorCV_scale_balance:
  def __init__(self, X, y):
    self.X = X
    self.y = y

  def evaluate_classifier_with_imputation(self, imputer=None, linear_flag=0, n_splits=5):
    """
    Evaluates classifiers using k-fold cross-validation and stores mean AUC.

    Args:
      imputer: Imputer object for handling missing values (optional).
      linear_flag: Flag to choose classifier (0: GBM, 1: LDA, 2: SVC, 3: RF).
      n_splits: Number of folds for k-fold cross-validation (default: 5).

    Returns:
      mean_auc: The average Area Under the ROC Curve (AUC) across folds.
      accuracies: A list of accuracy scores for each fold.
      class_type: String representing the evaluated classifier type.
    """

    classifiers = {
      0: (HistGradientBoostingClassifier(), 'GB'),
      1: (LinearDiscriminantAnalysis(), 'LDA'),
      2: (CalibratedClassifierCV(SVC(C=0.1, kernel='linear', probability=True, random_state=1)), 'SVC'),
      3: (RandomForestClassifier(max_depth=2, random_state=0), 'RF')
    }

    clf, class_type = classifiers[linear_flag]
    #kfold = KFold(n_splits=n_splits, shuffle=True)
    
    # Initialize StratifiedKFold
    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    
    aucs, accuracies = [], []

    for train_index, test_index in stratified_kfold.split(self.X, self.y):
      X_train, X_test = self.X[train_index], self.X[test_index]
      y_train, y_test = self.y[train_index], self.y[test_index]

      if (imputer is not None) & (linear_flag != 0):
          X_train_raw = imputer.fit_transform(X_train)
          X_test_raw = imputer.transform(X_test)
          scaler = StandardScaler()
          X_train= scaler.fit_transform(X_train_raw)
          X_test= scaler.transform(X_test_raw)

      if X_train.shape[0] < 6 or X_test.shape[0] < 6 or X_train.shape[1] < 4 or X_test.shape[1] < 4:
        auc = np.nan
        accuracy = np.nan
      else:
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]

        try:
          auc = roc_auc_score(y_test, y_prob)
        except ValueError as e:
          print(e)
          auc = None  # or choose a specific value

        accuracy = accuracy_score(y_test, np.round(y_prob))

      aucs.append(auc)
      accuracies.append(accuracy)

    mean_auc = np.mean(aucs)
    mean_acc = np.mean(accuracies)
    return mean_auc, mean_acc, class_type

class ModifiedClassifierEvaluatorCV_scale_balance:
  def __init__(self, X, y):
    self.X = X
    self.y = y

  def evaluate_classifier_with_imputation(self, imputer=None, linear_flag=0, n_splits=5):
    """
    Evaluates classifiers with imputation (before train/test split) and k-fold CV.

    Args:
      imputer: Imputer object for handling missing values (optional).
      linear_flag: Flag to choose classifier (0: GBM, 1: LDA, 2: SVC, 3: RF).
      n_splits: Number of folds for k-fold cross-validation (default: 5).

    Returns:
      mean_auc: The average Area Under the ROC Curve (AUC) across folds.
      accuracies: A list of accuracy scores for each fold.
      class_type: String representing the evaluated classifier type.
    """

    classifiers = {
      0: (HistGradientBoostingClassifier(), 'GB'),
      1: (LinearDiscriminantAnalysis(), 'LDA'),
      2: (CalibratedClassifierCV(SVC(C=0.1, kernel='linear', probability=True, random_state=1)), 'SVC'),
      3: (RandomForestClassifier(max_depth=2, random_state=0), 'RF')
    }

    clf, class_type = classifiers[linear_flag]
    #kfold = KFold(n_splits=n_splits, shuffle=True)
    # Initialize StratifiedKFold
    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    
    aucs, accuracies = [], []

    # Impute on the entire dataset before CV
    if imputer is not None:
      self.X = imputer.fit_transform(self.X)

    for train_index, test_index in stratified_kfold.split(self.X, self.y):
      X_train_raw, X_test_raw = self.X[train_index], self.X[test_index]
      y_train, y_test = self.y[train_index], self.y[test_index]
      scaler = StandardScaler()
      X_train= scaler.fit_transform(X_train_raw)
      X_test= scaler.transform(X_test_raw)

      if X_train.shape[0] < 6 or X_test.shape[0] < 6 or X_train.shape[1] < 4 or X_test.shape[1] < 4:
        auc = np.nan
        accuracy = np.nan
      else:
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]

        try:
          auc = roc_auc_score(y_test, y_prob)
        except ValueError as e:
          print(e)
          auc = None  # or choose a specific value

        accuracy = accuracy_score(y_test, np.round(y_prob))

      aucs.append(auc)
      accuracies.append(accuracy)

    mean_auc = np.mean(aucs)
    mean_acc = np.mean(accuracies)
    return mean_auc, mean_acc, class_type