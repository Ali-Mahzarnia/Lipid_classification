#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 11:37:50 2024

@author: amahzarn
"""

import os
os.chdir("/Users/amahzarn/Desktop/oct24/lipids_binary_PTB/")

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, regularizers

# Set random seeds for reproducibility
np.random.seed(42)  
tf.random.set_seed(42)  

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
    roc_auc_score_value = roc_auc_score(y_val, y_val_pred)  
    print(f'Trial {trial+1} AUC: {roc_auc_score_value:.4f}')

    auc_scores.append(roc_auc_score_value) 
    
    

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
