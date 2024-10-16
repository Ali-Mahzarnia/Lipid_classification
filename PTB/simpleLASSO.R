
library(pROC)
library(glmnet)

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


set.seed(234) # for consistency in train and test set selection
train.index = sample(1:length(y), size = floor(length(y)*0.8), replace = FALSE)



cvfit <- NULL
while(is.null(cvfit)) { 
  cvfit <- try(cv.glmnet(X[train.index,], y[train.index], 
                         family = "binomial", 
                         alpha = 1, 
                         penalty.factor = c(0, 0, 0, 0, 0, rep(1, ncol(X) - 5)), 
                         nfolds = 5), silent = TRUE)
  if(inherits(cvfit, "try-error")) {
    cvfit <- NULL  # Reset cvfit to NULL to retry
  }
  
}

png("RegPath_LASSO.png", width = 800, height = 600)
plot(cvfit)
dev.off()


model <- glmnet(X[train.index,], y[train.index], 
                family = "binomial", alpha = 1, lambda = cvfit$lambda.min, 
                penalty.factor = c(0, 0, 0, 0, 0, rep(1, ncol(X) - 5)) )

sum(coef(model)[-1]!=0) # number of non sparse features
paste(colnames(X)[which(coef(model)[-1]!=0)], collapse = ", ")


# Generate predicted probabilities for test set
prob <- predict(model, newx =X[-train.index,] , type = "response")
prob_vector <- prob[, 1]  

# Compute ROC curve
roc_curve <- roc(y[-train.index], prob_vector)

# Compute AUC
auc_value <- auc(roc_curve)
print(paste0("AUC:",  round(auc_value,3)))

# Compute confidence interval for AUC
auc_ci <- ci(roc_curve)
print(paste0("AUC CI:(", round(auc_ci[1],3), ",", round(auc_ci[3],3) , ")" ))


png("ROC_LASSO.png", width = 800, height = 600)
# Plot ROC curve
plot(roc_curve, main = "ROC Curve", col = "blue", lwd = 2)
dev.off()


#### CV whole data 
set.seed(234)
cvfit <- cvfit <- NULL
while(is.null(cvfit)) { 
  cvfit <- try(cv.glmnet(X, y , 
                         family = "binomial", 
                         alpha = 1, 
                         penalty.factor = c(0, 0, 0, 0, 0, rep(1, ncol(X) - 5)), 
                         nfolds = 5), silent = TRUE)
  if(inherits(cvfit, "try-error")) {
    cvfit <- NULL  # Reset cvfit to NULL to retry
  }
  
}


png("RegPath_LASSO_whole.png", width = 800, height = 600)
plot(cvfit)
dev.off()


model <- glmnet(X, y, family = "binomial", alpha = 1, 
                lambda = cvfit$lambda.min,
                penalty.factor = c(0, 0, 0, 0, 0, rep(1, ncol(X) - 5)) )
sum(coef(model)[-1]!=0) # number of non sparse features
paste0(colnames(X)[which(coef(model)[-1]!=0)], collapse = ", ")


# Generate predicted probabilities for test set
prob <- predict(model, newx =X , type = "response")
prob_vector <- prob[, 1]  

# Compute ROC curve
roc_curve <- roc(y, prob_vector)

# Compute AUC
auc_value <- auc(roc_curve)
print(paste0("AUC:",  round(auc_value,3)))

# Compute confidence interval for AUC
auc_ci <- ci(roc_curve)
print(paste0("AUC CI:(", round(auc_ci[1],3), ",", round(auc_ci[3],3) , ")" ))


png("ROC_LASSO_whole.png", width = 800, height = 600)
# Plot ROC curve
plot(roc_curve, main = "ROC Curve", col = "blue", lwd = 2)
dev.off()

