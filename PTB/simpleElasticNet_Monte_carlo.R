library(pROC)
library(glmnet)
library(parallel)
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
AUCs= rep(0,n_trials )
alphas = seq(0,1,by=0.1) # 11 alpha for CV
# To store the frequency of features being selected
feature_frequency = rep(0, ncol(X))
best_alphas = rep(0, n_trials)  # To store best alphas


set.seed(234) # for consistency in train and test set selection

for (trial in 1:n_trials){
  
train.index = sample(1:length(y), size = floor(length(y)*0.8), replace = FALSE)
if (any(apply(X[train.index, ], 2, var))) {train.index = sample(1:length(y), size = floor(length(y) * 0.8), replace = FALSE)}

cv_results <- mclapply(alphas, function(alpha) {
  result <- NULL
  while (is.null(result)) {
    tryCatch({
      # Run the cross-validation for the current alpha
      cvfit <- cv.glmnet(X[train.index,], y[train.index], family = "binomial", alpha = alpha, type.measure = 'auc', 
                         nfolds = 3, penalty.factor = c(rep(0,8), rep(1, ncol(X) - 8)), 
                         nlambda = 10, thresh = 1e-3)
      # Extract the minimum cross-validation error, max auc
      result <- max(cvfit$cvm)
      
    }, error = function(e) {
      result <- NULL  # Ensure the loop continues until no error
    })
  }
  return(result)
}, mc.cores = parallel::detectCores() - 5)



best_alpha <- alphas[which.max(cv_results)] # selecting best alpha
best_alphas[trial] = best_alpha

cvfit <- NULL
while (is.null(cvfit)) {
  tryCatch({
    # Run the cross-validation for the best alpha
    cvfit <- cv.glmnet(X[train.index,], y[train.index], family = "binomial", 
                       alpha = best_alpha, type.measure = 'auc' ,penalty.factor = c(rep(0,8), rep(1, ncol(X) - 8)), 
                       nfolds = 5)
    
  }, error = function(e) {
    cvfit <- NULL  # Ensure the loop continues until no error
  })
}


model <- glmnet(X[train.index,], y[train.index], family = "binomial", 
                alpha = best_alpha, lambda = cvfit$lambda.min, 
                penalty.factor = c(rep(0,8), rep(1, ncol(X) -8)) )


# Update feature frequency based on non-zero coefficients
selected_features <- coef(model) != 0
feature_frequency <- feature_frequency + as.numeric(selected_features[-1])  # Exclude intercept

# Generate predicted probabilities for test
prob <- predict(model, newx =X[-train.index,] , type = "response")
prob_vector <- prob[, 1]  

# Compute ROC curve
roc_curve <-suppressMessages(roc(y[-train.index], prob_vector))



# Compute AUC
auc_value <- auc(roc_curve)
AUCs[trial] <- auc_value
# report
cat("Trial", trial, " with AUC: " ,round(auc_value,3), "\n")

}



# Write AUC to Excel file
output_file <- "EN_Lipid_PTB_AUCs.xlsx"
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
output_file <- "EN_Lipid_features_PTB.xlsx"
write.xlsx(selected_features_data, file = output_file)





# Write best alphas to a separate Excel file
best_alphas_data <- data.frame(
  Trial = 1:n_trials,
  Best_Alpha = best_alphas
)
output_file_alphas <- "EN_Best_alphas_PTB.xlsx"
write.xlsx(best_alphas_data, file = output_file_alphas)
