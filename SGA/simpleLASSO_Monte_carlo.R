library(pROC)
library(glmnet)
library(xlsx)

data <- read.csv("Lipid_mom_SGA.csv")

data$Multiple.Birth[is.na(data$Multiple.Birth)] = 1
# Convert categorical variables to factors
data$Sex <- as.factor(data$Sex)
data$Multiple.Birth <- as.factor(data$Multiple.Birth)

# Create the design matrix with dummy variables
X_feature = as.matrix(data[, 11:(dim(data)[2]-1)]) # Except for SGA
age = data$AgeDelivery
bmi = data$BMI
GAUltrasound = data$GAUltrasound
multi_birth = data$Multiple.Birth
multi_birth[is.na(multi_birth)] = 1

# Create dummy variables
dummy_sex <- model.matrix(~ Sex - 1, data)  # Dummy variables for sex
dummy_multi_birth <- model.matrix(~ Multiple.Birth - 1, data)  # Dummy variables for multi_birth


#Combine features
y = data$SGA_weight
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

# Write AUC to Excel file
output_file <- "Lasso_Lipid_SGA_AUCs.xlsx"
write.xlsx(AUCs, file = output_file)


percentiles <- quantile(AUCs, probs = c(0.025, 0.975)) ## CI
cat("Average AUC:", mean(AUCs), "\n")
cat("with Percentile 95% CI of (", round(percentiles[1], 3), ",", round(percentiles[2], 3), ")\n")


mean_auc <- mean(AUCs) # mean of AUC
std_dev_auc <- sd(AUCs) # sd of AUCs
n <- length(AUCs) # number of AUCs
z_critical <- qnorm(0.975)  # 95% confidence interval
# Compute the margin of error and confidence interval
margin_error <- z_critical * (std_dev_auc / sqrt(n))
cat("Average AUC:", mean_auc, "\n95% Z-interval: (", round(mean_auc - margin_error, 3), ",", round(mean_auc + margin_error, 3), ")\n")


# Prepare data for Excel
feature_names <- colnames(X)  # Get feature names
selected_features_data <- data.frame(
  Feature = feature_names[feature_frequency > 0],
  Frequency = feature_frequency[feature_frequency > 0]
)
selected_features_data$mean_0=NA
selected_features_data$mean_1=NA

for (i in 1:nrow(selected_features_data)) {
  feature_name <- selected_features_data$Feature[i]
  j = which(colnames(X)==feature_name)
  # Compute average for control
  selected_features_data$mean_0[i] <- mean(as.numeric(X[y == 0, j]))
  # Compute average for case
  selected_features_data$mean_1[i] <- mean(as.numeric(X[y == 1, j]))
}

selected_features_data <- selected_features_data[order(selected_features_data$Frequency, decreasing = T), ]


# Write to Excel file
output_file <- "Lasso_Lipid_features_SGA.xlsx"
write.xlsx(selected_features_data, file = output_file)