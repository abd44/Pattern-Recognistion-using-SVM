#############################SVM Image recognition###################
################################################################
#Business Understanding
#Data Understanding
#Data Preparation & EDA
#Model Building 
#Model Evaluation
################################################################
#Business Understanding:We have an image of a digit submitted by a user via a scanner, a tablet, or other digital devices. 
#The goal is to develop a model that can correctly identify the digit (between 0-9) written in an image using SVM



#Install and load required Packages:
install.packages("caret")
install.packages("kernlab")
install.packages("dplyr")
install.packages("readr")
install.packages("ggplot2")
install.packages("gridExtra")

#Loading Neccessary libraries

library(kernlab)
library(readr)
library(caret)
library(readr)
library(caTools)
library(dplyr)
setwd("C:/Users/adubey53/Desktop/Abhi backup/Abhinav_backup/PGDDA/Course_4 ML/Assignment/SVM Dataset")

#Data Prepration & EDA

#Loading the data sets:
mnist_test <- read.csv("mnist_test.csv",header = F)
mnist_train <- read.csv("mnist_train.csv",header = F)

#Merging datasets with flag coulmn as 'test' and 'train' so as they can be separated,without disturbing original proportion of test and train data sets.
# The objective of merge is that we can clean data uniformly over test and train data and After data cleaning,we can seperate the test and train data retaining the original proportion .

mnist_test$flag <- "test"
mnist_train$flag <- "train"

#Binding the test and train data together so as same set of data cleaning can be performed:
mnist_data <- rbind(mnist_train,mnist_test)
dim(mnist_data)

#Checking missing value:Checking NA's for each coulmn and extracting rows which has NA's
colSums(is.na(mnist_data))
nrow(mnist_data[!(colSums(is.na(mnist_data)) == 0),])

# 0 row returned with NA's value.Hence we can conclude no NA's in data

#Removing Coulmn with redundant data,which has same value throughout.As the variables with the same data throughout has 0 variance and could be marked as redundant and can be dropped
mnist_data <- mnist_data[sapply(mnist_data, function(x) {!length(unique(x))== 1})]

#Checking Blanks:
mnist_data[mnist_data == ""]
#No Blank data

#Changing name of O/p Variable,i.e 1st coulmn and making it categorical type :

mnist_data$V1 <- as.factor(mnist_data$V1)
colnames(mnist_data)[1] <- 'Digits'


#Since, we have data processsed uniformly throughout train and test combined in a single dataset.
#Now,we can seperate the data with flag initially set up as "train" and "test",without disturbing the test data and train data quantity.

mnist_test <- mnist_data  %>% filter(flag == "test")

mnist_trainnn <- mnist_data  %>% filter(flag == "train")

mnist_trainnn$flag <- NULL 
mnist_test$flag <- NULL

#Since the train data set is so large.Taking 15% of the sample from train data.
set.seed(100)
sample_indices = sample(1:nrow(mnist_trainnn), .15*nrow(mnist_trainnn))
mnist_train_sample = mnist_trainnn[sample_indices,]
#Now we have mnist_train_sample as train sample data which would be easy to work on and will be less time consuming to prepare the model out of it.

ggplot(mnist_train_sample,aes(x=mnist_train_sample$Digits)) + geom_bar(stat = "count")

#All the Digits are uniformly distributed thus there will be no bias in the digits distrubution.Hence,we are good with the sample.

install.packages("doParallel")
library(doParallel)
cl <- makeCluster(detectCores())
registerDoParallel(cl)

#Model using vanilla dot

model_vanilla <- ksvm(Digits ~ ., data = mnist_train_sample,scale = FALSE,kernel="vanilladot")
evaluate_vanilla <- predict(model_vanilla,mnist_test)
confusionMatrix(evaluate_vanilla,mnist_test$Digits)


## Model Building

#--------------------------------------------------------------------
# Linear model - SVM  at Cost(C) = 1
#####################################################################

# Model with C =1
model_1<- ksvm(Digits ~ ., data = mnist_train_sample,scale = FALSE,C=1)

# Predicting the model results 
evaluate_1<- predict(model_1, mnist_test)

# Confusion Matrix - Finding accuracy, Sensitivity and specificity
confusionMatrix(evaluate_1, mnist_test$Digits)


#With Linear Model got Accuracy of 95.58% with C=1,Hence getting decent accuracy in Linear model means that data is well linearly seperable


# Using RBF Kernel Method:
Model_RBF <- ksvm(Digits~ ., data = mnist_train_sample, scale = FALSE, kernel = "rbfdot")
Eval_RBF<- predict(Model_RBF, mnist_test)

#confusion matrix - RBF Kernel
confusionMatrix(Eval_RBF,mnist_test$Digits)
Model_RBF

#Accuracy of the RBF Model is 0.955,ie 95% of the digits are getting classified correctly.
# Parameter cost is C=1 and hyperparameter: sigma = 1.631e-07
#Even with very less sigma value means that not more of non-linearity is present in the data,can be classified linearly

############   Hyperparameter tuning and Cross Validation #####################

# Using train function from caret package to perform Cross Validation: 

#traincontrol function Controls the computational nuances of the train function.
# i.e. method =  CV means  Cross Validation.
#      Number = 5 implies Number of folds in CV.

trainControl <- trainControl(method="cv", number=5)

# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.

metric <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.
#From RBF model we got Sigma as 1.63e-07.So trying tuning sigma value around the same .
#cost C we observerd as 1,so tuning the value of C around value 1.
set.seed(100)
grid <- expand.grid(.sigma=c(0.50e-07,1.63e-07,2.7e-07,3e-07), .C=c(0.5,1,2,3) )

fit.svm <- train(Digits~., data=mnist_train_sample, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl,allowParallel=TRUE)
print(fit.svm)
plot(fit.svm)

#The final values used for the model were sigma = 3e-07 and C = 3 with Accuracy of 96.79%,after the tuning of Hyperparameter tuning and cross validation


######################################################################
# Checking overfitting - Non-Linear - SVM
######################################################################

# Validating the model results on test data
evaluate_non_linear<- predict(fit.svm, mnist_test)
confusionMatrix(evaluate_non_linear, mnist_test$Digits)

#Accuracy of the model on test data is 96.74 % & accuracy on training model is 96.79% which is a good classifier of digits for a unknown test data.
#


###Now Since we have tuned Hyperparameter after cross validation which is:
#sigma = 3.e-07 and C = 3. We can use it for RBF model to see the Accuracy of the model.

#RBF model using tuned parameters:
Model_RBF_tuned <- ksvm(Digits~ ., data = mnist_train_sample, scale = FALSE, kernel = "rbfdot",
                        C=3,kpar=list(sigma=3e-07))

Eval_RBF_tuned<- predict(Model_RBF_tuned, mnist_test)
#confusion matrix - Tuned RBF Kernel: 
confusionMatrix(Eval_RBF_tuned,mnist_test$Digits)
Model_RBF_tuned

#Got Accuracy as 96.74%,Using RBF Model on test data with Tuned parameters such as :Hyperparameter : sigma =  3e-07 & cost C = 3.

#Thus,RBF Model is working very well on handwritten digit recognition with the parameters that we got using cross validation and hypertuning.
#Hence, RBF Model is working pretty well in classifying hand written digits on test data with the tuned parameters.



