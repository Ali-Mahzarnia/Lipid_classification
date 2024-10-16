library(pROC)
library(glmnet)
library(xlsx)

data <- read.csv("Lipid_mom_PTB.csv")

data$Multiple.Birth[is.na(data$Multiple.Birth)] = 1
# Convert categorical variables to factors
data$Sex <- as.factor(data$Sex)
data$Multiple.Birth <- as.factor(data$Multiple.Birth)

# Create the design matrix with dummy variables
X_feature = as.matrix(data[, 11:(dim(data)[2]-1)]) # Except for PTB
age = data$AgeDelivery
bmi = data$BMI
GAUltrasound = data$GAUltrasound
multi_birth = data$Multiple.Birth
multi_birth[is.na(multi_birth)] = 1

# Create dummy variables
dummy_sex <- model.matrix(~ Sex - 1, data)  # Dummy variables for sex
dummy_multi_birth <- model.matrix(~ Multiple.Birth - 1, data)  # Dummy variables for multi_birth


#Combine features
y = data$PTB
X = cbind(age, bmi, dummy_sex, dummy_multi_birth, GAUltrasound, X_feature)  # Combine all features

n_trials = 100
AUCs= rep(0, n_trials)

# To store the frequency of features being selected
feature_frequency = rep(0, ncol(X))

set.seed(234) # for consistency in train and test set selection

for (trial in 1:n_trials) {
  train.index = sample(1:length(y), size = floor(length(y) * 0.8), replace = FALSE)
  
  # Handle convergence error in cv.glmnet
  cvfit <- NULL
  while (is.null(cvfit)) { 
    cvfit <- try(cv.glmnet(X[train.index, ], y[train.index], 
                           family = "binomial", 
                           alpha = 1, 
                           penalty.factor = c(rep(0,8), rep(1, ncol(X) - 8)) , type.measure = 'auc', 
                           nfolds = 5), silent = TRUE)
    if (inherits(cvfit, "try-error")) {
      cvfit <- NULL  # Reset cvfit to NULL to retry
    }
  }
  
  # Run with best lambda
  model <- glmnet(X[train.index, ], y[train.index], family = "binomial", 
                  alpha = 1, lambda = cvfit$lambda.min, 
                  penalty.factor = c(rep(0,8), rep(1, ncol(X) - 8)))
  
  # Update feature frequency based on non-zero coefficients
  selected_features <- coef(model) != 0
  feature_frequency <- feature_frequency + as.numeric(selected_features[-1])  # Exclude intercept
  
  # Generate predicted probabilities for test set
  prob <- predict(model, newx = X[-train.index, ], type = "response")
  prob_vector <- prob[, 1]  
  # Compute ROC curve
  roc_curve <- suppressMessages(roc(y[-train.index], prob_vector))
  # Compute AUC
  auc_value <- auc(roc_curve)
  AUCs[trial] <- auc_value
  # Report
  cat("Trial", trial, " with AUC: ", round(auc_value, 3), "\n")
}

percentiles <- quantile(AUCs, probs = c(0.025, 0.975)) ## CI
cat("Average AUC:", mean(AUCs), "\n")
cat("with 95% CI of (", round(percentiles[1], 3), ",", round(percentiles[2], 3), ")\n")

# Prepare data for Excel
feature_names <- colnames(X)  # Get feature names
selected_features_data <- data.frame(
  Feature = feature_names[feature_frequency > 50],
  Frequency = feature_frequency[feature_frequency > 50]
)

# Write to Excel file
output_file <- "Lasso_Lipid_features_50.xlsx"
write.xlsx(selected_features_data, file = output_file)