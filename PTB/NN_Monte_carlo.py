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
from scipy import stats


# Set random seeds for reproducibility
np.random.seed(42)  
tf.random.set_seed(42)  

# Load the data from a CSV file
data = pd.read_csv('Lipid_mom_PTB.csv')

# Extract BMI, Sex, AgeDelivery, GAUltrasound, Multiple.Birth, columns, and separate them from the regular features
data['Multiple.Birth'] = data['Multiple.Birth'].fillna(1)
bmi_sex_age_cols = ['BMI', 'Sex', 'AgeDelivery', 'GAUltrasound', 'Multiple.Birth']
X_bmi_sex_age = data[bmi_sex_age_cols].values  # Features we don't want to penalize
# Assuming X_bmi_sex_age is a NumPy array
X_bmi_sex_age = pd.DataFrame(X_bmi_sex_age, columns=bmi_sex_age_cols)
# Convert 'Sex' and 'multi_birth' into dummy variables
X_bmi_sex_age = pd.get_dummies(X_bmi_sex_age, columns=['Sex', 'Multiple.Birth'], drop_first=True)

# Select remaining features (from column 11 to the last column, excluding target column)
X_rest = data.iloc[:, 10:-1].values  # Regular features to be penalized
y = data.iloc[:, -1].values  # Target: PTB



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
def create_model(input_shape_rest, input_shape_non_penalized):
    # Separate inputs for penalized and non-penalized features
    inputs_rest = layers.Input(shape=input_shape_rest)  # Penalized features
    inputs_non_penalized = layers.Input(shape=input_shape_non_penalized)  # Non-penalized features

    # Penalized features go through Dense layers with regularization
   
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs_rest)  
    x = layers.Dropout(0.5)(x)  
    
    
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)  
    x = layers.Dropout(0.5)(x)  
    
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)  
    x = layers.Dropout(0.5)(x)  
    
    # Concatenate penalized and non-penalized features
    combined = layers.concatenate([x, inputs_non_penalized])  

    # Attention Layer
    attention_output = AttentionLayer()(combined)  

    # Flatten for Dense layer
    attention_output = layers.Flatten()(attention_output)  

    # Output layer for binary classification
    outputs = layers.Dense(1, activation='sigmoid')(attention_output)  

    # Create the model
    model = models.Model(inputs=[inputs_rest, inputs_non_penalized], outputs=outputs)
    return model


# Number of trials
n_trials = 1000
auc_scores = []


# Perform the experiment 100 times
for trial in range(n_trials):
    
    # Split the data into training and validation sets (80% for training, 20% for validation)
    X_rest_train, X_rest_val, X_bmi_sex_age_train, X_bmi_sex_age_val, y_train, y_val = train_test_split( X_rest, X_bmi_sex_age, y, test_size=0.2, random_state=trial)
    # Normalize the penalized features (not BMI, Sex, AgeDelivery)
    combined_train = np.concatenate([X_rest_train, X_bmi_sex_age_train], axis=1)
    scaler = StandardScaler()
    combined_train_scaled = scaler.fit_transform(combined_train)
    X_rest_train_scaled = combined_train_scaled[:, :X_rest_train.shape[1]]  # First part is X_rest_train
    X_bmi_sex_age_train_scaled = combined_train_scaled[:, X_rest_train.shape[1]:]  # Second part is X_bmi_sex_age_train
    
    
    # Combine the validation sets temporarily
    combined_val = np.concatenate([X_rest_val, X_bmi_sex_age_val], axis=1)

    # Transform the validation set using the already fitted scaler
    combined_val_scaled = scaler.transform(combined_val)

    # Separate the scaled validation set
    X_rest_val_scaled = combined_val_scaled[:, :X_rest_val.shape[1]]
    X_bmi_sex_age_val_scaled = combined_val_scaled[:, X_rest_val.shape[1]:]
    
    
    # Corrected input shapes (ensure they are tuples)
    input_shape_rest = (X_rest_train.shape[1],)  # Shape of penalized features (tuple)
    input_shape_non_penalized = (X_bmi_sex_age_train.shape[1],)  # Shape of non-penalized features (tuple)
    
    # Create and compile the model
    model = create_model(input_shape_rest, input_shape_non_penalized)  
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    
    
    # Train the model
    model.fit([X_rest_train_scaled, X_bmi_sex_age_train_scaled], y_train, epochs=50, batch_size=32, validation_split=0.2, verbose =0 )
    
    # Evaluate the model
    y_val_pred = model.predict([X_rest_val_scaled, X_bmi_sex_age_val_scaled]).flatten()  
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




# Calculate the mean and standard deviation of AUC scores
mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores, ddof=1)  # Use ddof=1 for sample standard deviation

# Number of trials
n_trials = len(auc_scores)

# Compute the Z-critical value for a 95% confidence interval (Z_critical = 1.96)
z_critical = stats.norm.ppf(0.975)

# Compute the margin of error
margin_of_error = z_critical * (std_auc / np.sqrt(n_trials))

# Compute the confidence interval
ci_low = mean_auc - margin_of_error
ci_high = mean_auc + margin_of_error

# Report the results
print(f"Mean AUC over {n_trials} trials: {mean_auc:.4f}")
print(f"95% Confidence Interval for AUC (Z-interval): ({ci_low:.4f}, {ci_high:.4f})")
