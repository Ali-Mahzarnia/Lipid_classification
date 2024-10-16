
library(pROC)
library(glmnet)
library(parallel)

data <- read.csv("Lipid_mom_PTB.csv")
X_feature = as.matrix(data[,11:(dim(data)[2]-1)]) # Except for age
age = data$AgeDelivery
bmi = data$BMI
sex = data$Sex
GAUltrasound = data$GAUltrasound
multi_birth = data$Multiple.Birth  
multi_birth[is.na(multi_birth)] = 1

y = data$PTB


X = cbind(age,bmi,sex,GAUltrasound, multi_birth,X_feature) # combine the age and features


n_trials = 1000
AUCs= rep(0,n_trials )
alphas = seq(0,1,by=0.1) # 11 alpha for CV

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
