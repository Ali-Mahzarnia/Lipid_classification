#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 11:37:50 2024

@author: amahzarn
"""

import os
os.chdir("/Users/amahzarn/Desktop/oct24/lipids_binary_sga/")

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, regularizers

import numpy as np
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve




# Set random seeds for reproducibility
np.random.seed(42)  
tf.random.set_seed(42)  

# Read the dataset
data = pd.read_csv('Lipid_mom_SGA.csv')

# Select the relevant columns (columns 11 to second last + AgeDelivery, BMI, Sex)
X = data.iloc[:, 10:-1].copy()  # Columns 11 to the one before the last column
X['AgeDelivery'] = data['AgeDelivery']
X['BMI'] = data['BMI']
X['Sex'] = data['Sex']
X['GAUltrasound'] = data['GAUltrasound']
X['multi_birth'] = data['Multiple.Birth'].fillna(1)



# Define the target column 
y = data['SGA_weight']


# Custom Attention Layer definition
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.dense_attention = layers.Dense(1)  # Dense layer to compute attention scores

    def call(self, inputs):
        attention_scores = self.dense_attention(inputs)  
        attention_scores = tf.squeeze(attention_scores, axis=-1)  

        attention_weights = tf.nn.softmax(attention_scores, axis=-1)  

        attention_weights = tf.expand_dims(attention_weights, axis=-1)  

        weighted_input = layers.multiply([inputs, attention_weights])  

        return weighted_input  

# Build the model function
def create_model(input_shape_rest):
    # Separate inputs for penalized and non-penalized features
    inputs_rest = layers.Input(shape=input_shape_rest)  # Penalized features

    # Penalized features go through Dense layers with regularization
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs_rest)  
    x = layers.Dropout(0.5)(x)  
    
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)  
    x = layers.Dropout(0.5)(x)  
    
    # Attention Layer
    attention_output = AttentionLayer()(x)  

    # Flatten for Dense layer
    attention_output = layers.Flatten()(attention_output)  

    # Output layer for binary classification
    outputs = layers.Dense(1, activation='sigmoid')(attention_output)  

    # Create the model
    model = models.Model(inputs=[inputs_rest], outputs=outputs)
    return model



    
# Step 4: Split the data into 80% training and 20% validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Step 1: Standardize the feature set (X_train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_val = scaler.transform(X_val)  


# Corrected input shapes (ensure they are tuples)
input_shape_rest = (X_train_scaled.shape[1],)  # Shape of penalized features (tuple)

# Create and compile the model
model = create_model(input_shape_rest)  
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])


# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose =0 )

# Evaluate the model
y_val_pred = model.predict(X_val).flatten() 
 
y_test_true = np.array(y_val)
y_test_pred_prob = np.array(y_val_pred)

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
plt.savefig('NN_penal_all_roc_curve_train_set.png', dpi=300)
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


# Corrected input shapes (ensure they are tuples)
input_shape_rest = (X_scaled.shape[1],)  # Shape of penalized features (tuple)

# Create and compile the model
model = create_model(input_shape_rest)  
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])


# Train the model
model.fit(X_scaled, y, epochs=50, batch_size=32, validation_split=0.2, verbose =0 )

# Evaluate the model
y_pred = model.predict(X_scaled).flatten() 
 
y_true = np.array(y)
y_pred_prob = np.array(y_pred)


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
plt.savefig('NN_penal_All_roc_curve_cv.png', dpi=300)
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

print(f"AUC: {auc_test_score:.4f}")
print(f"95% CI for AUC: ({confidence_lower:.4f} , {confidence_upper:.4f})")













