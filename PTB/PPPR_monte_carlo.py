#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:05:43 2024

@author: amahzarn
"""

import os
os.chdir("/Users/amahzarn/Desktop/oct24/PPRisk/PTB/")

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from metric_learn import NCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
import numpy as np
import scipy.stats as st

# Read the dataset
data = pd.read_csv('Lipid_mom_PTB.csv')

# Select the relevant columns (columns 11 to second last + AgeDelivery, BMI, Sex)
X = data.iloc[:, 10:-1].copy()  # Columns 11 to the one before the last column
X['AgeDelivery'] = data['AgeDelivery']
X['BMI'] = data['BMI']
X['Sex'] = data['Sex']
X['GAUltrasound'] = data['GAUltrasound']
X['multi_birth'] = data['Multiple.Birth'].fillna(1)



# Define the target column (PTB_weight)
y = data['PTB']

# Number of trials
n_trials = 1000
auc_scores = []

# Perform the experiment 100 times
for trial in range(n_trials):
    # Step 4: Split the data into 80% training and 20% validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=trial)


    # Step 1: Standardize the feature set (X_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Step 2: Initialize and fit the NCA model
    nca = NCA(max_iter=1000, random_state=42)
    nca.fit(X_train_scaled, y_train)

    # Step 3: Transform the training and validation data using the learned NCA metric
    X_train_nca = nca.transform(X_train_scaled)
    X_val_scaled = scaler.transform(X_val)
    X_val_nca = nca.transform(X_val_scaled)

    # Step 4: Separate the training set by cases (PTB_weight == 1) and controls (PTB_weight == 0)
    case_indices = np.where(y_train == 1)[0]
    control_indices = np.where(y_train == 0)[0]

    # Initialize lists to store true labels and predictions for the validation set
    y_test_true = []
    y_test_pred_prob = []

    # Step 5: For each validation sample, find the 10% closest cases and controls
    for i in range(X_val_nca.shape[0]):
        distances = distance.cdist(X_val_nca[i].reshape(1, -1), X_train_nca, 'euclidean').flatten()

        # Find the 10% closest cases and controls
        case_distances = distances[case_indices]
        control_distances = distances[control_indices]
        n_closest_cases = int(0.1 * len(case_distances))
        n_closest_controls = int(0.1 * len(control_distances))

        closest_case_idx = case_indices[np.argsort(case_distances)[:n_closest_cases]]
        closest_control_idx = control_indices[np.argsort(control_distances)[:n_closest_controls]]

        # Combine the closest cases and controls
        closest_samples_idx = np.concatenate([closest_case_idx, closest_control_idx])

        # Get the corresponding data and labels for the closest samples
        X_closest_samples = X_train_nca[closest_samples_idx]
        y_closest_samples = y_train.iloc[closest_samples_idx]

        # Fit sparse logistic regression on the closest samples
        logreg = LogisticRegression(penalty='l1', solver='liblinear', random_state=42,class_weight='balanced')
        logreg.fit(X_closest_samples, y_closest_samples)

        # Predict the probability for the current validation sample
        y_prob = logreg.predict_proba(X_val_nca[i].reshape(1, -1))[:, 1]

        # Store the true label and predicted probability
        y_test_true.append(y_val.iloc[i])
        y_test_pred_prob.append(y_prob[0])

    # Convert the lists to numpy arrays
    y_test_true = np.array(y_test_true)
    y_test_pred_prob = np.array(y_test_pred_prob)

    # Compute the test set AUC for this trial
    auc_test_score = roc_auc_score(y_test_true, y_test_pred_prob)
    auc_scores.append(auc_test_score)
    print(f"Trial {trial+1} Test set AUC: {auc_test_score:.4f}")

# Convert the list of AUC scores to a numpy array
auc_scores = np.array(auc_scores)

# Calculate the mean AUC
mean_auc = np.mean(auc_scores)

# Calculate the 95% confidence interval using percentiles (non-parametric method)
ci_low = np.percentile(auc_scores, 2.5)
ci_high = np.percentile(auc_scores, 97.5)

# Report the results
print(f"Mean AUC over {n_trials} trials: {mean_auc:.4f}")
print(f"95% Confidence Interval for AUC: ({ci_low:.4f}, {ci_high:.4f})")
