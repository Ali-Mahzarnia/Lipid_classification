
library(pROC)
library(glmnet)
library(parallel)

data <- read.csv("Lipid_mom_SGA.csv")
X_feature = as.matrix(data[,11:(dim(data)[2]-1)]) # Except for age
age = data$AgeDelivery
bmi = data$BMI
sex = data$Sex
GAUltrasound = data$GAUltrasound
multi_birth = data$Multiple.Birth  
multi_birth[is.na(multi_birth)] = 1

y = data$SGA_weight


X = cbind(age,bmi,sex,GAUltrasound, multi_birth,X_feature) # combine the age and features


n_trials = 100
AUCs= rep(0,n_trials )
alphas = seq(0,1,by=0.1) # 11 alpha for CV
# To store the frequency of features being selected
feature_frequency = rep(0, ncol(X))
best_alphas = rep(0, n_trials)  # To store best alphas


set.seed(234) # for consistency in train and test set selection

for (trial in 1:n_trials){
  
train.index = sample(1:length(y), size = floor(length(y)*0.8), replace = FALSE)

cv_results <- mclapply(alphas, function(alpha) {
  result <- NULL
  while (is.null(result)) {
    tryCatch({
      # Run the cross-validation for the current alpha
      cvfit <- cv.glmnet(X[train.index,], y[train.index], family = "binomial", alpha = alpha, 
                         nfolds = 3, penalty.factor = c(0, 0, 0, 0, 0, rep(1, ncol(X) - 5)), 
                         nlambda = 10, thresh = 1e-3)
      # Extract the minimum cross-validation error
      result <- min(cvfit$cvm)
      
    }, error = function(e) {
      result <- NULL  # Ensure the loop continues until no error
    })
  }
  return(result)
}, mc.cores = parallel::detectCores() - 5)



best_alpha <- alphas[which.min(cv_results)] # selecting best alpha
best_alphas[trial] = best_alpha

cvfit <- NULL
while (is.null(cvfit)) {
  tryCatch({
    # Run the cross-validation for the best alpha
    cvfit <- cv.glmnet(X[train.index,], y[train.index], family = "binomial", 
                       alpha = best_alpha, penalty.factor = c(0, 0, 0, 0, 0, rep(1, ncol(X) - 5)), 
                       nfolds = 5)
    
  }, error = function(e) {
    cvfit <- NULL  # Ensure the loop continues until no error
  })
}


model <- glmnet(X[train.index,], y[train.index], family = "binomial", 
                alpha = best_alpha, lambda = cvfit$lambda.min, 
                penalty.factor = c(0, 0, 0, 0, 0, rep(1, ncol(X) - 5)) )


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

percentiles <- quantile(AUCs, probs = c(0.025, 0.975)) ## CI
cat("Average AUC:", mean(AUCs), "\n")
cat("with 95% CI of (" , round(percentiles[1],3), ",", round(percentiles[2],3),")" )




# Prepare data for Excel
feature_names <- colnames(X)  # Get feature names
selected_features_data <- data.frame(
  Feature = feature_names[feature_frequency > 70],
  Frequency = feature_frequency[feature_frequency > 70]
)

# Write to Excel file
output_file <- "EN_Lipid_features_70.xlsx"
write.xlsx(selected_features_data, file = output_file)

# Write best alphas to a separate Excel file
best_alphas_data <- data.frame(
  Trial = 1:n_trials,
  Best_Alpha = best_alphas
)
output_file_alphas <- "Best_alphas.xlsx"
write.xlsx(best_alphas_data, file = output_file_alphas)
