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

import numpy as np
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


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


# Step 4: Split the data into 80% training and 20% validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of the splits
print("Training set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_val.shape, y_val.shape)


# Distance metric leanring
# Step 1: Standardize the feature set (X_train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Step 2: Initialize the NCA model
nca = NCA(max_iter=1000, random_state=42)

# Step 3: Fit the NCA model to the training data
nca.fit(X_train_scaled, y_train)

# Step 4: Transform the training data using the learned NCA metric
X_train_nca = nca.transform(X_train_scaled)

# Step 5: Transform the validation set similarly (using the scaler and NCA transformation)
X_val_scaled = scaler.transform(X_val)
X_val_nca = nca.transform(X_val_scaled)

# Check the shape of the transformed training data
print("NCA Transformed Training set shape:", X_train_nca.shape)
print("NCA Transformed Validation set shape:", X_val_nca.shape)

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

# Compute the test set AUC 
auc_test_score = roc_auc_score(y_test_true, y_test_pred_prob)
print(f" Test set AUC: {auc_test_score:.4f}")


# Step 8: Compute ROC curve
fpr, tpr, _ = roc_curve(y_test_true, y_test_pred_prob)

# Step 9: Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc_test_score:.4f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Save the figure with 300 ppi
plt.savefig('roc_curve_train_set.png', dpi=300)
plt.show()

n_bootstraps = 1000
rng_seed = 42  # random seed for reproducibility
bootstrapped_scores = []


# Set random seed
rng = np.random.RandomState(rng_seed)

# Bootstrapping
for i in range(n_bootstraps):
    indices = rng.randint(0, len(y_test_true), len(y_test_true))
    if len(np.unique(y_test_true[indices])) < 2:
        # Skip this bootstrap sample because it only contains one class
        continue
    
    score = roc_auc_score(y_test_true[indices], y_test_pred_prob[indices])
    bootstrapped_scores.append(score)

# Calculate the confidence interval
sorted_scores = np.array(bootstrapped_scores)
sorted_scores.sort()

# 95% confidence interval (can change to 90% or other)
confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

print(f"AUC: {auc_test_score:.4f}")
print(f"95% CI for AUC: ({confidence_lower:.4f} , {confidence_upper:.4f})")



########### CV


# Distance metric leanring
# Step 1: Standardize the feature set (X_train)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Initialize the NCA model
nca = NCA(max_iter=1000, random_state=42)

# Step 3: Fit the NCA model to the training data
nca.fit(X_scaled, y)

# Step 4: Transform the training data using the learned NCA metric
X_nca = nca.transform(X_scaled)


# Step 4: Separate the training set by cases (PTB_weight == 1) and controls (PTB_weight == 0)
case_indices = np.where(y == 1)[0]
control_indices = np.where(y == 0)[0]

# Initialize lists to store true labels and predictions for the validation set
y_true = []
y_pred_prob = []

# Step 5: For each validation sample, find the 10% closest cases and controls
for i in range(X_nca.shape[0]):
    distances = distance.cdist(X_nca[i].reshape(1, -1), X_nca, 'euclidean').flatten()

    # Find the 10% closest cases and controls
    case_distances = distances[case_indices]
    control_distances = distances[control_indices]
    n_closest_cases = int(0.1 * len(case_distances))
    n_closest_controls = int(0.1 * len(control_distances))

    closest_case_idx = case_indices[np.argsort(case_distances)[:n_closest_cases]]
    closest_control_idx = control_indices[np.argsort(control_distances)[:n_closest_controls]]

    # Combine the closest cases and controls
    closest_samples_idx = np.concatenate([closest_case_idx, closest_control_idx])
    closest_samples_idx = closest_samples_idx [closest_samples_idx!=i]
    # Get the corresponding data and labels for the closest samples
    X_closest_samples = X_nca[closest_samples_idx]
    y_closest_samples = y.iloc[closest_samples_idx]

    # Fit sparse logistic regression on the closest samples
    logreg = LogisticRegression(penalty='l1', solver='liblinear', random_state=42,class_weight='balanced')
    logreg.fit(X_closest_samples, y_closest_samples)

    # Predict the probability for the current validation sample
    y_prob = logreg.predict_proba(X_nca[i].reshape(1, -1))[:, 1]

    # Store the true label and predicted probability
    y_true.append(y.iloc[i])
    y_pred_prob.append(y_prob[0])

# Convert the lists to numpy arrays
y_true = np.array(y_true)
y_pred_prob = np.array(y_pred_prob)

# Compute the test set AUC 
auc_score = roc_auc_score(y_true, y_pred_prob)
print(f" Test set AUC: {auc_score:.4f}")


# Step 8: Compute ROC curve
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)

# Step 9: Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Save the figure with 300 ppi
plt.savefig('roc_curve_cv.png', dpi=300)
plt.show()

n_bootstraps = 1000
rng_seed = 42  # random seed for reproducibility
bootstrapped_scores = []


# Set random seed
rng = np.random.RandomState(rng_seed)

# Bootstrapping
for i in range(n_bootstraps):
    indices = rng.randint(0, len(y_true), len(y_true))
    if len(np.unique(y_true[indices])) < 2:
        # Skip this bootstrap sample because it only contains one class
        continue
    
    score = roc_auc_score(y_true[indices], y_pred_prob[indices])
    bootstrapped_scores.append(score)

# Calculate the confidence interval
sorted_scores = np.array(bootstrapped_scores)
sorted_scores.sort()

# 95% confidence interval (can change to 90% or other)
confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

print(f"95% CI for AUC: ({confidence_lower:.4f} , {confidence_upper:.4f})")











