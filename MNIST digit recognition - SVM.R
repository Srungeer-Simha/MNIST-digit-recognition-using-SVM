############################################## Loading libraries ###########################################

library(ggplot2)
library(kernlab)
library(caret)
library(caTools)
library(gridExtra)


############################################## Business objective ##########################################

# To succesfully classify handwritten digits (0-9) using pixel values 
# Support Vector Machines will be applied


################################################# Loading data #############################################

mnist_train <- read.csv("mnist_train.csv", stringsAsFactors = F, header = F)
mnist_test <- read.csv("mnist_test.csv", stringsAsFactors = F, header = F)

View(mnist_train) # Data has no column names
View(mnist_test) # Data has no column names

names(mnist_test)[1] <- "label"
names(mnist_train)[1] <- "label"


################################## Data cleaning, preparation & understanding ##############################

#--------------------------------------------- Data cleaning ----------------------------------------------#

## Checking for missing values, unnecessary rows and columns

# headers and footers

head(mnist_test, 1) # no unnecessary headers
head(mnist_train, 1) # no unnecessary headers

tail(mnist_test, 1) # no unnecessary footers
tail(mnist_train, 1) # no unnecessary footers

# Duplicated rows

sum(duplicated(mnist_test)) # no duplicate rows
sum(duplicated(mnist_train)) # no duplicate rows

# Checking for NAs

sum(sapply(mnist_test, function(x) sum(is.na(x)))) # There are no missing values
sum(sapply(mnist_train, function(x) sum(is.na(x)))) # There are no missing values


#------------------------------------------- Data understanding -------------------------------------------#

# The MNIST database of handwritten digits has a training set of 60,000 examples, 
# and a test set of 10,000 examples. It is a subset of a larger set available from NIST.
# The 784 columns apart from the label consist of  28*28 matrix describing the scanned image of the digits
# The digits have been size-normalized and centered in a fixed-size image

str(mnist_test) # all dependant variables are integers, 60000 observations, 785 variables
str(mnist_train) # all dependant variables integers, 5000 observations, 785 variables

summary(mnist_test[ , 2:100]) # some columns seem to be containing only zeros, Pixel values go upto 255,
summary(mnist_train[ , 2:100]) # but some only go up to ~100, data needs to be scaled


#-------------------------------------------- Data preparation --------------------------------------------#

# Convert label variable into factor

mnist_train$label <- factor(mnist_train$label)
summary(mnist_train$label)

mnist_test$label <- factor(mnist_test$label)
summary(mnist_test$label)

# Sampling training dataset

dim(mnist_train) # computation time would be unnaceptable for such a large dataset

set.seed(100)
sample_indices <- sample(1: nrow(mnist_train), 5000) # extracting subset of 5000 samples for modelling
train <- mnist_train[sample_indices, ]

# Scaling data 

max(train[ ,2:ncol(train)]) # max pixel value is 255, lets use this to scale data
train[ , 2:ncol(train)] <- train[ , 2:ncol(train)]/255

test <- cbind(label = mnist_test[ ,1], mnist_test[ , 2:ncol(mnist_test)]/255)


#----------------------------------------- Exploratory Data Analysis --------------------------------------#


## Distribution of digits across all data sets

plot1 <- ggplot(mnist_train, aes(x = label, y = (..count..)/sum(..count..))) + geom_bar() + theme_light() +
                labs(y = "Relative frequency", title = "mnist_train dataset") + 
                scale_y_continuous(labels=scales::percent, limits = c(0 , 0.15)) +
                geom_text(stat = "count", 
                          aes(label = scales:: percent((..count..)/sum(..count..)), vjust = -1))

plot2 <- ggplot(train, aes(x = label, y = (..count..)/sum(..count..))) + geom_bar() + theme_light() +
                labs(y = "Relative frequency", title = "train dataset") + 
                scale_y_continuous(labels=scales::percent, limits = c(0 , 0.15)) +
                geom_text(stat = "count", 
                          aes(label = scales:: percent((..count..)/sum(..count..)), vjust = -1))

plot3 <- ggplot(test, aes(x = label, y = (..count..)/sum(..count..))) + geom_bar() + theme_light() +
                labs(y = "Relative frequency", title = "test dataset") + 
                scale_y_continuous(labels=scales::percent, limits = c(0 , 0.15)) +
                geom_text(stat = "count", 
                          aes(label = scales:: percent((..count..)/sum(..count..)), vjust = -1))

grid.arrange(plot1, plot2, plot3, nrow = 3)

# Relative frequencies of the digits has been retained while sampling to create the reduced train data set
# Similar frequency in test dataset also observed


######################################### Model Building & Evaluation ######################################

#--------------------------------------------- Linear Kernel ----------------------------------------------#

## Linear kernel using default parameters

model1_linear <- ksvm(label ~ ., data = train, scaled = FALSE, kernel = "vanilladot", C = 1)
print(model1_linear) 

eval1_linear <- predict(model1_linear, newdata = test, type = "response")
confusionMatrix(eval1_linear, test$label) 

# Observations:
# Overall accuracy of 91.3%
# Specificities quite high > 99%
# Sensitivities good > 84%


## Linear kernel using stricter C

model2_linear <- ksvm(label ~ ., data = train, scaled = FALSE, kernel = "vanilladot", C = 10)
print(model2_linear) 

eval2_linear <- predict(model2_linear, newdata = test, type = "response")
confusionMatrix(eval2_linear, test$label) 

# Observations:
# Overall accuracy of 91%
# Model performance has slightly decreased, model may be overfitting


## Using cross validation to optimise C

grid_linear <- expand.grid(C= c(0.001, 0.1 ,1 ,10 ,100)) # defining range of C

fit.linear <- train(label ~ ., data = train, metric = "Accuracy", method = "svmLinear",
                    tuneGrid = grid_linear, preProcess = NULL,
                    trControl = trainControl(method = "cv", number = 5))

# printing results of 5 cross validation
print(fit.linear) 
plot(fit.linear)

# Observations:
# Best accuracy of 92% at C = 0.1
# Higher values of C are overfitting and lower values are too generic

eval_cv_linear <- predict(fit.linear, newdata = test)
confusionMatrix(eval_cv_linear, test$label)

# Observations:
# Overall accuracy of 92.4%, slightly imporved
# Specificities quite high > 99%
# Sensitivities > 86%, improved from model1 by making model more generic i.e. lower C 


#--------------------------------------------- Radial Kernel ----------------------------------------------#

## Radial kernel using default parameters

model1_rbf <- ksvm(label ~ ., data = train, scaled = FALSE, kernel = "rbfdot", C = 1, kpar = "automatic")
print(model1_rbf) 

eval1_rbf <- predict(model1_rbf, newdata = test, type = "response")
confusionMatrix(eval1_rbf, test$label) 

# Observations:
# Overall accuracy of 95%
# Specificities quite high > 99%
# Sensitivities high > 92%
# Increase in overall accuracy and sensitivty from linear kernel using C = 1, sigma = 0.0107
# data seems to have non linearity to it


## Redial kernel with higher sigma

model2_rbf <- ksvm(label ~ ., data = train, scaled = FALSE, kernel = "rbfdot",
                    C = 1, kpar = list(sigma = 1))
print(model2_rbf) 

eval2_rbf <- predict(model2_rbf, newdata = test, type = "response")
confusionMatrix(eval2_rbf, test$label) 

# Observations:
# Accuracy drops to 11% and class wise results are very poor
# sigma = 1 is too much non linearity and the model is overfitting


## Using cross validation to optimise C and sigma

# defining ranges of C and sigma
grid_rbf = expand.grid(C= c(0.01, 0.1, 1, 5, 10), sigma = c(0.001, 0.01, 0.1, 1, 5)) 

# Using only 2 folds to optimise run time
fit.rbf <- train(label ~ ., data = train, metric = "Accuracy", method = "svmRadial",tuneGrid = grid_rbf,
                 trControl = trainControl(method = "cv", number = 2), preProcess = NULL)

# printing results of 2 cross validation
print(fit.rbf) 
plot(fit.rbf)

# Observations:
# Best sigma value is ~ 0.01
# Higher sigma values are overfitting and lower sigma values are not capturing non linearity adequately
# Accuracy increases with C until 5 and then decreases again, can be further optimised

# Optimising C further
grid_rbf = expand.grid(C= c(1,2, 3, 4, 5, 6 ,7, 8, 9, 10), sigma = 0.01)

fit.rbf2 <- train(label ~ ., data = train, metric = "Accuracy", method = "svmRadial",tuneGrid = grid_rbf,
                     trControl = trainControl(method = "cv", number = 5), preProcess = NULL)

# printing results of cross validation
print(fit.rbf2) 
plot(fit.rbf2)

eval_cv_rbf <- predict(fit.rbf2, newdata = test)
confusionMatrix(eval_cv_rbf, test$label)

# Observations:
# Accuracy is highest at C = 3 and sigma = 0.01
# Higher C values are overfitting and lower C values have too much bias
# Accuracy of 96%
# High Sensitivities > 92%
# Very High Specificities > 99%


#--------------------------------------------- Linear Kernel ----------------------------------------------#

## Polynomial kernel with degree 2 and default scale and offset
model1_poly <- ksvm(label ~ ., data = train, kernel = "polydot", scaled = FALSE, C = 1, 
                    kpar = list(degree = 2, scale = 1, offset = 1))
print(model1_poly)

eval1_poly <- predict(model1_poly, newdata = test)
confusionMatrix(eval1_poly, test$label)

# Observations
# Good accuracy of 95.24%
# High Sensitivities > 92% and specificities > 99%
# Similar performance to radial kernel


## Polynomial kernel with varied scale
model2_poly <- ksvm(label ~ ., data = train, kernel = "polydot", scaled = FALSE, C = 1, 
                    kpar = list(degree = 2, scale = -2, offset = 1))
print(model2_poly)

eval2_poly <- predict(model2_poly, newdata = test)
confusionMatrix(eval2_poly, test$label)

# Observations
# Slight reduction in accuracy but similar perfromance


## Polynomial kernel with varied offset
model3_poly <- ksvm(label ~ ., data = train, kernel = "polydot", scaled = FALSE, C = 1, 
                    kpar = list(degree = 2, scale = 1, offset = 10))
print(model3_poly)

eval3_poly <- predict(model3_poly, newdata = test)
confusionMatrix(eval3_poly, test$label)

# Observations
# similar perfromance as before, scale and offset seem to have little effect on performance


## Polynomial kernel with higher C
model4_poly <- ksvm(label ~ ., data = train, kernel = "polydot", scaled = FALSE, C = 3, 
                    kpar = list(degree = 2, scale = 1, offset = 1))
print(model4_poly)

eval4_poly <- predict(model4_poly, newdata = test)
confusionMatrix(eval4_poly, test$label)

# Observations
# similar perfromance as before


## Grid search to optimise hyperparameters

grid_poly = expand.grid(C= c(0.01, 0.1, 1, 10), degree = c(1, 2, 3, 4, 5), 
                        scale = c(-100, -10, -1, 1, 10, 100))

fit.poly <- train(label ~ ., data = train, metric = "Accuracy", method = "svmPoly",tuneGrid = grid_poly,
                  trControl = trainControl(method = "cv", number = 2), preProcess = NULL)

# printing results of cross validation
print(fit.poly) 
plot(fit.poly)

eval_cv_poly <- predict(fit.poly, newdata = test)
confusionMatrix(eval_cv_poly, test$label)

# Observations:
# Best model obtained for C = 0.01, degree = 2, scale = 1
# as data has been scaled already scale = 1 is optimum
# C has little to no effect on perfomance, C = 0.01 generic model has been picked as optimum
# degrees higher than 2 are overfitting
# Accuracy of 95.24%, sensitivities > 92%, specificities > 99%


## Implementing optmised polynomial model 
model5_poly <- ksvm(label ~ ., data = train, kernel = "polydot", scaled = FALSE, C = 0.01, 
                    kpar = list(degree = 2, scale = 1, offset = 0.5))
print(model5_poly)

eval5_poly <- predict(model5_poly, newdata = test)
confusionMatrix(eval5_poly, test$label)

# Observations:
# offset of 0.5 used as independent variables are in the range of 0 to 1
# best accuracy of polynomial kernels 95.25%


################################################ Conclusion ################################################

# Final model
final_model = fit.rbf2

## SVM using RBF kernel (C = 3, sigma = 0.01) achieved highest accuracy in predicting digits

# reduced training data set of 5000 instances (extracted using random sampling) has been used
    # distribution of the dependent variable (digtits) has been preserved while sampling
# Model performance on validation data set of 10000 instances
    # Accuracy = 95.46%
    # Sensitivites > 92%
    # Specificities > 99%
# Polynomial kernel (C = 0.01, degree = 2, scale = 1, offset = 0.05) also perfromed very well
    # performance metrics are only marginally lesser than radial kernel
    # Run time is better than that of radial kernel
