First we load the relevant libraries and datasets for this assignment:

    # Import libraries
    library(ggplot2)
    library(dplyr)
    library(class)
    library(gridExtra)
    library(gmodels)
    library(caret)
    library(corrplot)
    library(clusterGeneration)
    library(mnormt)
    library(mosaic)
    library(tidyverse)
    library(FNN)
    library(broom)
    library(glmnet)
    library(knitr)

    # Import datasets
    SClass <- read.csv("~/Github/SDS323_Spring2020/sclass.csv")
    News <- read.csv("~/Github/SDS323_Spring2020/online_news.csv")
    data(SaratogaHouses)

    # Set random seed for entire markdown document
    set.seed(346)

    # Define function for calculating root mean squared error (RMSE)
    rmse = function(y, yhat) {
      sqrt( mean( (y - yhat)^2 ) )
    }

Problem 1: Mercedes S Class Price Prediction
============================================

Problem Statement and Background
--------------------------------

In this problem, we attempt to use a dataset that contains information
on over 29,000 Mercedes S class vehicles to predict the price of the car
based on only the mileage, for each of the two following trims
separately: (1) 350 and (2) 63 AMG. The objective is to build a
K-nearest neighbors (KNN) model that performs the best based on
out-of-sample root mean-squared error (RMSE). Consequently, for each of
the two subsets of the original dataset, we need to find the optimal
number of neighbors, or k value.

Methods: Setting Up the Experimental Framework for Optimizing the KNN Model
---------------------------------------------------------------------------

First, we isolate two subsets of the dataset based on the 350 and 63 AMG
trims. Then we use the Caret package to implement leave-one-out
cross-validation (LOOCV) on both datasets since there is only one
predictive feature and a relatively small number of observations. This
will allow us to yield a better estimate of the performances of the
model variants in each subset.

    # Isolate subsets of original dataset by trim 
    S350 <- subset(SClass, trim == 350)
    S63A <- subset(SClass, trim == '63 AMG')

    # Isolate only the relevant variables
    S350 <- S350[c("price","mileage")]
    S63A <- S63A[c("price","mileage")]

    # Specify using LOOCV during training
    trControl <- trainControl(method  = "LOOCV")

### Results: Optimal K-Value in KNN Model is Higher for 63 AMG Trims than 350 Trims

We see that for the subset containing the 350 trims, the best k-value
was 20, giving a RMSE of about 9980. In contrast, the subset containing
the 63 AMG trim has the optimal k-value of 65, giving a RMSE of about
14400 The subset with the 63 AMG trim has a larger optimal k-value.

    # Fit KNN variants to S350 dataset
    S350fit <- train(price ~ .,
                 method     = "knn",
                 tuneGrid   = expand.grid(k = 1:30),
                 trControl  = trControl,
                 metric     = "RMSE",
                 data       = S350)

    # Plot results
    ggplot(S350fit, aes(x = k, y = RMSE)) +
      geom_point() + 
      ylab("RMSE") +
      xlab("Number of Neighbors (k)") +
      labs(title = "KNN Performance by Number of Neighbors (k)", subtitle = "For 350 Trims Only",caption = "RMSE from LOOCV. Error bars supressed for clarity.")

![](Exercise02_files/figure-markdown_strict/unnamed-chunk-3-1.png)


    # Fit KNN variants to S63A dataset
    S63Afit <- train(price ~ .,
                 method     = "knn",
                 tuneGrid   = expand.grid(k = c(1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100)),
                 trControl  = trControl,
                 metric     = "RMSE",
                 data       = S63A[c("price","mileage")])

    # Plot results
    ggplot(S63Afit, aes(x = k, y = RMSE)) +
      geom_point() + 
      ylab("RMSE") +
      xlab("Number of Neighbors (k)") +
      labs(title = "KNN Performance by Number of Neighbors (k)", subtitle = "For 63 AMG Trims Only", caption = "RMSE from LOOCV. Error bars supressed for clarity.")

![](Exercise02_files/figure-markdown_strict/unnamed-chunk-3-2.png)

### Conclusion: 63 AMG Trim Model Has a Higher K Value Due to Denser Data Points

The reason why the 63 AMG trim requites a higher k-value is most likely
due to the fact that there are more data points in the 63 AMG
sub-dataset than that of the 350 sub-dataset. Consequently, the data
points in the 63 AMG sub-dataset are "denser" and may require more
neighbors to ease out the noise from irregular data points.
Additionally, more neighbors may be used without inducing underfitting
due to the higher density of points.

Problem 2: Saratoga Houses
==========================

Problem Statement and Background
--------------------------------

In this problem, we use the SaratogaHouses dataset to attempt to create
a model that predicts the value of a house, given a set of features such
as the number of bedrooms and land value. In particular, we will explore
different variants of linear regression and KNN models to examine the
performance of the two models relative to each other. Starting with a
preliminary model, We examine whether the reduction of a single feature
results in model improvement, and whether the addition of a single
composite feature results in model improvement. The ultimate objective
of this problem is to provide a price-modeling strategy for the local
taxing authority.

Methods and Results: Designing a Linear Regression Model
--------------------------------------------------------

Given that linear regression is a fairly simple and fast model to train,
we have the luxury of somewhat "brute-forcing" the way we "hand-craft"
our final regression model, which will be extremely apparent in
subsequent subsections.

### Initial Model Prototype

In the start R script (saratoga\_lm.R), the four models were implemented
based on the following features shown below, along with an new model
that we shall work from (Model 5):

-   Model V1: lotSize, bedrooms, bathrooms
-   Model V2: All variables excluding sewer, waterfront, landvalue,
    newConstruction
-   Model V3: All composite variables not including sewer, waterfront,
    landvalue, newConstruction
-   Model V4: lotSize, age, livingArea, pctCollege, bedrooms,
    fireplaces, bathrooms, rooms, heating, fuel, centralAir,
    lotSize:heating + livingArea:rooms + newConstruction,
    livingArea:newConstruction
-   Model V5: All variables with no additional adjustments

We easily see that a linear regression model that makes use of all
available variables in the dataset significantly outperforms the other
four baseline models covered in class.

    # Find size of train and test sets for a 80/20 split
    n = nrow(SaratogaHouses)
    n_train = round(0.8*n)  
    n_test = n - n_train

    rmse_vals = do(100)*{
      
      # Re-split into train and test sets
      train_cases = sample.int(n, n_train, replace=FALSE)
      test_cases = setdiff(1:n, train_cases)
      saratoga_train = SaratogaHouses[train_cases,]
      saratoga_test = SaratogaHouses[test_cases,]
      
      # LR model variants
      lm1 = lm(price ~ lotSize + bedrooms + bathrooms, data=saratoga_train)
      lm2 = lm(price ~ . - sewer - waterfront - landValue - newConstruction, data=saratoga_train)
      lm3 = lm(price ~ (. - sewer - waterfront - landValue - newConstruction)^2, data=saratoga_train)
      lm_dominate = lm(price ~ lotSize + age + livingArea + pctCollege + bedrooms + fireplaces + bathrooms + 
                       rooms + heating + fuel + centralAir + lotSize:heating + livingArea:rooms + newConstruction +
                       livingArea:newConstruction, data=saratoga_train)
      lm_all = lm(price~ ., data=saratoga_train)
      
      # Predictions out of sample
      yhat_test1 = predict(lm1, saratoga_test)
      yhat_test2 = predict(lm2, saratoga_test)
      yhat_test3 = predict(lm3, saratoga_test)
      yhat_test4 = predict(lm_dominate, saratoga_test)
      yhat_test5 = predict(lm_all, saratoga_test)
      
      # Store RMSE values
      c(rmse(saratoga_test$price, yhat_test1),
        rmse(saratoga_test$price, yhat_test2),
        rmse(saratoga_test$price, yhat_test3),
        rmse(saratoga_test$price, yhat_test4),
        rmse(saratoga_test$price, yhat_test5))
    }

    # Calculate mean model performance and plot results
    colMeans(rmse_vals)
          V1       V2       V3       V4       V5 
    77147.60 66280.68 71728.88 66183.95 58649.71 
    boxplot(rmse_vals, ylab = "RMSE", outline=FALSE, las = 2, names = c("Class 1"," Class 2", "Class 3", "Class 4", "Proposed"))
    title("Linear Regression Model Variants")

![](Exercise02_files/figure-markdown_strict/unnamed-chunk-4-1.png)

### Backward Feature Selection

Given that the model with all features performs the best, a plausible
next step would be removing one feature at a time and seeing if any of
these models perform better. We see that for the most part, removing a
particular variable does not seem to improve the model, and in some
cases even worsen the performance. One unsurprising exception is when we
remove the feature "landValue" or "livingArea", performance
significantly worsens. Given this outcome, we shall make the naive
conclusion that the removal of any features will not result in further
improvement of the model. It is worth noting that in the case of large
datasets, a model with less features with the same performance is more
desirable since it will be leaner and train and operate faster. Since
our dataset is small and our model (linear regression) is simple, this
is not as large of a concern.

    # Find size of train and test sets for a 80/20 split
    n = nrow(SaratogaHouses)
    n_train = round(0.8*n)  
    n_test = n - n_train

    rmse_vals = do(100)*{
      
      # Re-split into train and test sets
      train_cases = sample.int(n, n_train, replace=FALSE)
      test_cases = setdiff(1:n, train_cases)
      saratoga_train = SaratogaHouses[train_cases,]
      saratoga_test = SaratogaHouses[test_cases,]
      
      # LR model variants
      lm_00 = lm(price ~ ., data=saratoga_train)
      lm_01 <- lm(price ~ . - lotSize, data=saratoga_train)
      lm_02 <- lm(price ~ . - age, data=saratoga_train)
      lm_03 <- lm(price ~ . - landValue, data=saratoga_train)
      lm_04 <- lm(price ~ . - livingArea, data=saratoga_train)
      lm_05 <- lm(price ~ . - pctCollege, data=saratoga_train)
      lm_06 <- lm(price ~ . - bedrooms, data=saratoga_train)
      lm_07 <- lm(price ~ . - fireplaces, data=saratoga_train)
      lm_08 <- lm(price ~ . - bathrooms, data=saratoga_train)
      lm_09 <- lm(price ~ . - rooms, data=saratoga_train)
      lm_10 <- lm(price ~ . - heating, data=saratoga_train)
      lm_11 <- lm(price ~ . - fuel, data=saratoga_train)
      lm_12 <- lm(price ~ . - sewer, data=saratoga_train)
      lm_13 <- lm(price ~ . - waterfront, data=saratoga_train)
      lm_14 <- lm(price ~ . - newConstruction, data=saratoga_train)
      lm_15 <- lm(price ~ . - centralAir, data=saratoga_train)
      
      # Predictions out of sample
      yhat_test00 = predict(lm_00, saratoga_test)
      yhat_test01 = predict(lm_01, saratoga_test)
      yhat_test02 = predict(lm_02, saratoga_test)
      yhat_test03 = predict(lm_03, saratoga_test)
      yhat_test04 = predict(lm_04, saratoga_test)
      yhat_test05 = predict(lm_05, saratoga_test)
      yhat_test06 = predict(lm_06, saratoga_test)
      yhat_test07 = predict(lm_07, saratoga_test)
      yhat_test08 = predict(lm_08, saratoga_test)
      yhat_test09 = predict(lm_09, saratoga_test)
      yhat_test10 = predict(lm_10, saratoga_test)
      yhat_test11 = predict(lm_11, saratoga_test)
      yhat_test12 = predict(lm_12, saratoga_test)
      yhat_test13 = predict(lm_13, saratoga_test)
      yhat_test14 = predict(lm_14, saratoga_test)
      yhat_test15 = predict(lm_15, saratoga_test)
      
      # Store RMSE values
      c(rmse(saratoga_test$price, yhat_test00),
        rmse(saratoga_test$price, yhat_test01),
        rmse(saratoga_test$price, yhat_test02),
        rmse(saratoga_test$price, yhat_test03),
        rmse(saratoga_test$price, yhat_test04),
        rmse(saratoga_test$price, yhat_test05),
        rmse(saratoga_test$price, yhat_test06),
        rmse(saratoga_test$price, yhat_test07),
        rmse(saratoga_test$price, yhat_test08),
        rmse(saratoga_test$price, yhat_test09),
        rmse(saratoga_test$price, yhat_test10),
        rmse(saratoga_test$price, yhat_test11),
        rmse(saratoga_test$price, yhat_test12),
        rmse(saratoga_test$price, yhat_test13),
        rmse(saratoga_test$price, yhat_test14),
        rmse(saratoga_test$price, yhat_test15))
    }

    # Calculate mean model performance and plot results
    colMeans(rmse_vals)
          V1       V2       V3       V4       V5       V6       V7       V8 
    59011.67 59222.20 59042.96 65137.34 62710.74 58987.21 59138.82 58965.16 
          V9      V10      V11      V12      V13      V14      V15      V16 
    59777.04 59155.21 59073.38 58979.90 58892.05 60027.44 59606.56 59106.48 
    boxplot(rmse_vals, ylab = "RMSE", outline=FALSE, las = 2, names = c("Baseline","lotSize","age","landValue","livingArea","pctCollege","bedrooms","fireplaces","bathrooms","rooms","heating","fuel","sewer","waterfront","newConstruction","centralAir"))
    title("Linear Regression Model Variants")

![](Exercise02_files/figure-markdown_strict/unnamed-chunk-5-1.png)

### Evaluating Composite Features

Currently, our model uses all features and the reduction of any lone
feature does not appear to improve the model. Consequently, a new
direction that we can take is by trying to incorporate composite
features. This is useful because in some sense, it allows the regression
model to behave "nonlinearly" in the weakest and loosest definition.
With 15 different features, this allows for 105 possible combinations of
pair-wise composite features. Naturally, a forward feature selection
method with these new composite features may seem like a reasonable
choice, but it would take a long time to test all 105 new model variants
without significant computing power. Perhaps instead, we can limit our
composite features to pairs in which the two features seem related. For
example, "bedrooms:bathrooms" may be a reasonable composite feature,
whereas "bedrooms:sewer" may not make logical sense. Of course, the best
model will not necessary yield from the "most logical" composite
features, but it will be a simple heuristic that we will follow to limit
our search range. Consequently, we will examine how the addition of the
following new composite features will affect our regression model:

-   V1: All features
-   V2: livingArea:bedrooms
-   V3: livingArea:bathrooms
-   V4: livingArea:rooms
-   V5: bedrooms:bathrooms
-   V6: bedrooms:rooms
-   V7: bathrooms:rooms
-   V8: heating:fuel
-   V9: heating:centralAir

<!-- -->

    # Find size of train and test sets for a 80/20 split
    n = nrow(SaratogaHouses)
    n_train = round(0.8*n)  
    n_test = n - n_train

    rmse_vals = do(100)*{
      
      # Re-split into train and test sets
      train_cases = sample.int(n, n_train, replace=FALSE)
      test_cases = setdiff(1:n, train_cases)
      saratoga_train = SaratogaHouses[train_cases,]
      saratoga_test = SaratogaHouses[test_cases,]
      
      # LR model variants
      lm_00 = lm(price ~ ., data=saratoga_train)
      lm_01 = lm(price ~ . + livingArea:bedrooms, data=saratoga_train)
      lm_02 = lm(price ~ . + livingArea:bathrooms, data=saratoga_train)
      lm_03 = lm(price ~ . + livingArea:rooms, data=saratoga_train)
      lm_04 = lm(price ~ . + bedrooms:bathrooms, data=saratoga_train)
      lm_05 = lm(price ~ . + bedrooms:rooms, data=saratoga_train)
      lm_06 = lm(price ~ . + bathrooms:rooms, data=saratoga_train)
      lm_07 = lm(price ~ . + heating:fuel, data=saratoga_train)
      lm_08 = lm(price ~ . + heating:centralAir, data=saratoga_train)
      
      # Predictions out of sample
      yhat_test00 = predict(lm_00, saratoga_test)
      yhat_test01 = predict(lm_01, saratoga_test)
      yhat_test02 = predict(lm_02, saratoga_test)
      yhat_test03 = predict(lm_03, saratoga_test)
      yhat_test04 = predict(lm_04, saratoga_test)
      yhat_test05 = predict(lm_05, saratoga_test)
      yhat_test06 = predict(lm_06, saratoga_test)
      yhat_test07 = predict(lm_07, saratoga_test)
      yhat_test08 = predict(lm_08, saratoga_test)
      
      # Store RMSE values
      c(rmse(saratoga_test$price, yhat_test00),
        rmse(saratoga_test$price, yhat_test01),
        rmse(saratoga_test$price, yhat_test02),
        rmse(saratoga_test$price, yhat_test03),
        rmse(saratoga_test$price, yhat_test04),
        rmse(saratoga_test$price, yhat_test05),
        rmse(saratoga_test$price, yhat_test06),
        rmse(saratoga_test$price, yhat_test07),
        rmse(saratoga_test$price, yhat_test08))
    }

    # Calculate mean model performance and plot results
    colMeans(rmse_vals)
          V1       V2       V3       V4       V5       V6       V7       V8 
    58736.51 58825.12 58622.85 58707.40 58814.58 58775.96 58655.93 58809.06 
          V9 
    58716.59 
    boxplot(rmse_vals, ylab = "RMSE", outline=FALSE, las = 2, names = c("Baseline","livingArea:bedrooms","livingArea:bathrooms","livingArea:rooms","bedrooms:bathrooms","bedrooms:rooms","bathrooms:rooms","heating:fuel","heating:centralAir"))
    title("Linear Regression Model Variants")

![](Exercise02_files/figure-markdown_strict/unnamed-chunk-6-1.png)

Again, it seems that the addition of these new features - in this case
composite features - does not seem to have a significant effect on model
performance. Perhaps we should settle with our current regression model
that just uses all existing features without any polynomial terms.
Normally, this would be a poor excuse to stop developing a model, as
there are so many other possible options to try out, such as using
transformations or polynomial terms. We justify this however, by noting
that we will also attempt to turn the regression model into a k-nearest
neighbors model. A k-nearest-neighbor model is adaptable enough to find
certain complex relationships, such as polynomial terms and other simple
interactions. The k-nearest-neighbor model is unable to find composite
features, but our limited analysis suggests that we are not missing out
on much with that regard anyways.

Methods and Results: Development of a K-Nearest Neighbor Model
--------------------------------------------------------------

### Creating a Fair Comparison Between Linear Regression and K-Nearest Neighbor Model

It is worth noting that KNN models cannot handle categorical data unless
it is one-hot encoded. For binary features, such as "waterfront",
"newConstruction" and "centralAir", this can be easily done by
converting "Yes" to 1 and "No" to 0. For categorical features with more
than two classes, this can expand the number of features due to one-hot
encoding. Consequently, we will remove the features "heating", "fuel",
and "sewer". To make this a fair comparison however, we should compare
it with a linear regression model that also removes these features.
Conveniently, a linear regression model without the aforementioned three
features performs similarly to a linear regression model that uses all
features. Consequently, we can easily compare the linear regression and
KNN models without fear that the removal of these three features may
make the model less effective.

    # Find size of train and test sets for a 80/20 split
    n = nrow(SaratogaHouses)
    n_train = round(0.8*n)  
    n_test = n - n_train

    rmse_vals = do(100)*{
      
      # Re-split into train and test sets
      train_cases = sample.int(n, n_train, replace=FALSE)
      test_cases = setdiff(1:n, train_cases)
      saratoga_train = SaratogaHouses[train_cases,]
      saratoga_test = SaratogaHouses[test_cases,]
      
      # LR model variants
      lm_0 = lm(price~ ., data=saratoga_train)
      lm_1 = lm(price~ . - heating - fuel - sewer, data=saratoga_train)
      
      # Predictions out of sample
      yhat_test0 = predict(lm_0, saratoga_test)
      yhat_test1 = predict(lm_1, saratoga_test)
      
      # Store RMSE values
      c(rmse(saratoga_test$price, yhat_test0),
        rmse(saratoga_test$price, yhat_test1))
    }

    # Calculate mean model performance and plot results
    colMeans(rmse_vals)
          V1       V2 
    58939.23 58934.71 
    boxplot(rmse_vals, xlab = "Model Variant", ylab = "RMSE", outline=FALSE, names = c("All Features","Reduced Features"))
    title("Linear Regression Model Variants")

![](Exercise02_files/figure-markdown_strict/unnamed-chunk-7-1.png)

### Testing Different K-Nearest Neighbor Model Variants

Before using the k-nearest neighbor model, we standardize all numerical
features, with the exception of "price" since it is not being used as a
model input. We repeatedly, randomly split the train and test sets to be
80% and 20% of the total dataset a hundred times, respectively. Then we
display the results on a box plot with the outliers supressed. We also
calculate the mean RMSE of each model variant. We see the the best
performing KNN model is when k = 10, with a RMSE of about 61781.

    # Scale SaratogaHouses dataset
    ScaledSaratogaHouses <- SaratogaHouses
    ScaledSaratogaHouses <- transform(ScaledSaratogaHouses, waterfront = ifelse(waterfront == "Yes", 1, 0))
    ScaledSaratogaHouses <- transform(ScaledSaratogaHouses, newConstruction = ifelse(newConstruction == "Yes", 1, 0))
    ScaledSaratogaHouses <- transform(ScaledSaratogaHouses, centralAir = ifelse(centralAir == "Yes", 1, 0))
    ScaledSaratogaHouses[, -c(1,11:13)] <- scale(ScaledSaratogaHouses[, -c(1,11:13)])

    # Find size of train and test sets for a 80/20 split
    n = nrow(SaratogaHouses)
    n_train = round(0.8*n)  
    n_test = n - n_train

    rmse_vals = do(100)*{
      
      # Re-split into train and test sets
      train_cases = sample.int(n, n_train, replace=FALSE)
      test_cases = setdiff(1:n, train_cases)
      saratoga_train = ScaledSaratogaHouses[train_cases,]
      saratoga_test = ScaledSaratogaHouses[test_cases,]
      
      # KNN model variants
      KNN01 <- knn.reg(train = saratoga_train[-c(1,11:16)], test = saratoga_test[-c(1,11:16)], y = saratoga_train[c(1)], k = 1)
      KNN02 <- knn.reg(train = saratoga_train[-c(1,11:16)], test = saratoga_test[-c(1,11:16)], y = saratoga_train[c(1)], k = 3)
      KNN03 <- knn.reg(train = saratoga_train[-c(1,11:16)], test = saratoga_test[-c(1,11:16)], y = saratoga_train[c(1)], k = 5)
      KNN04 <- knn.reg(train = saratoga_train[-c(1,11:16)], test = saratoga_test[-c(1,11:16)], y = saratoga_train[c(1)], k = 10)
      KNN05 <- knn.reg(train = saratoga_train[-c(1,11:16)], test = saratoga_test[-c(1,11:16)], y = saratoga_train[c(1)], k = 15)
      KNN06 <- knn.reg(train = saratoga_train[-c(1,11:16)], test = saratoga_test[-c(1,11:16)], y = saratoga_train[c(1)], k = 25)
      KNN07 <- knn.reg(train = saratoga_train[-c(1,11:16)], test = saratoga_test[-c(1,11:16)], y = saratoga_train[c(1)], k = 50)
      
      # Store RMSE values
      c(rmse(saratoga_test$price, KNN01$pred),
        rmse(saratoga_test$price, KNN02$pred),
        rmse(saratoga_test$price, KNN03$pred),
        rmse(saratoga_test$price, KNN04$pred),
        rmse(saratoga_test$price, KNN05$pred),
        rmse(saratoga_test$price, KNN06$pred),
        rmse(saratoga_test$price, KNN07$pred))
    }

    # Calculate mean model performance and plot results
    colMeans(rmse_vals)
          V1       V2       V3       V4       V5       V6       V7 
    76351.84 64130.85 62644.96 61781.40 62189.27 62321.79 64384.06 
    boxplot(rmse_vals, xlab = "Model Variant", ylab = "RMSE", outline=FALSE, names = c("k = 1","k = 3","k = 5","k = 10","k = 15","k = 25","k = 50"))
    title("K-Nearest Neighbors Model Variants")

![](Exercise02_files/figure-markdown_strict/unnamed-chunk-8-1.png)

Conclusion: Report for Local Taxing Authority
---------------------------------------------

It appears that linear regression models perform slightly better than
KNN models. Consequently, using linear regression models instead of KNN
models would yield more accurate predictions for the local taxing
authority. Additionally, linear regression models yield the additional
benefit that it is much more interpretable than KNN models. For example,
the linear regression model allows you to easily conclude that "given
all other features equal, a change of X in the feature A would result in
a change of Y in house price". In contrast, a KNN model reasons
predictions somewhat akin to the argument "house A is more similar to
these other five houses, which has a price of Y, therefore house A has a
price near Y as well". This interpretability from linear regression
models may be particularly valuable should any residents go to court to
dispute the house value in an attempt to get a lower house value for
subsequently lower property tax. From a slightly more technical
viewpoint, the linear regression model is more robust than the KNN model
to noise, suggesting that there is quite a bit of fluctuation in house
prices in Saratoga on the basis of the available features in the
dataset.

Problem 3: Mashable Viral Articles
==================================

Problem Statement and Background
--------------------------------

The dataset we will work with contains all online articles published by
Mashable during 2013 and 2014. We wish to predict whether or not an
article is "viral" on the basis that a viral article has been shared
over 1400 times. We attempt to approach this problem through a variety
of modeling methods, such as linear regression and KNN. Additionally, we
will examine whether or not predicting the number of shares through
regression then subsequent classification (ie. "regression-first")
performs better than creating a new binary feature for viral or not from
the beginning and creating a classification model (ie.
"classification-first").

Methods: Feature Scaling, Dimension Reduction, and Feature Selection
--------------------------------------------------------------------

We remove the non-useful feature "url" from our dataset and then scale
all other features to be centered around zero with a standard deviation
of 1. While scaling is not critical for linear regression, it is crucial
for KNN models to allow for even treatment of each feature when
calculating distance measurements. With the scaled dataset, we then
implement principal component analysis (PCA) to reduce the number of
features from 36 features to 20 principal components. As shown in the
PCA analysis, the top 20 principal components capture about 84.45% of
the total dataset's variance with respect to the number of shares. By
choosing only the top number of principal components, we also
effectively perform feature selection by ignoring principal components
that capture minimal variance. Lastly, we introduce a new feature called
"Viral", in which 1 indicates the article is viral with over 1400
shares, and 0 indicates otherwise.

    # Remove "url" feature and scale remaining features
    News.Scaled <- scale(News[, -c(1,38)])
    News.Scaled <- as.data.frame(News.Scaled)

    # Perform principal component analysis and isolate the top 20 principal components
    News.pca <- prcomp(News.Scaled, center = TRUE, scale. = TRUE)
    News.PCA <- News.pca$x[,1:20]
    News.PCA <- as.data.frame(News.PCA)
    News.PCA$shares <- News$shares
    summary(News.pca)
    Importance of components:
                              PC1     PC2     PC3    PC4     PC5     PC6     PC7
    Standard deviation     1.9318 1.56473 1.48888 1.4186 1.39873 1.36045 1.19368
    Proportion of Variance 0.1037 0.06801 0.06158 0.0559 0.05435 0.05141 0.03958
    Cumulative Proportion  0.1037 0.17168 0.23325 0.2892 0.34350 0.39491 0.43449
                              PC8     PC9    PC10    PC11    PC12   PC13   PC14
    Standard deviation     1.1665 1.13345 1.11045 1.10850 1.09724 1.0882 1.0849
    Proportion of Variance 0.0378 0.03569 0.03425 0.03413 0.03344 0.0329 0.0327
    Cumulative Proportion  0.4723 0.50798 0.54223 0.57636 0.60981 0.6427 0.6754
                              PC15    PC16    PC17    PC18    PC19    PC20    PC21
    Standard deviation     1.06647 1.05612 1.02512 0.98945 0.97456 0.92388 0.88421
    Proportion of Variance 0.03159 0.03098 0.02919 0.02719 0.02638 0.02371 0.02172
    Cumulative Proportion  0.70699 0.73798 0.76717 0.79436 0.82074 0.84445 0.86617
                             PC22    PC23    PC24    PC25    PC26    PC27    PC28
    Standard deviation     0.8507 0.81907 0.76751 0.72655 0.69217 0.66228 0.63270
    Proportion of Variance 0.0201 0.01864 0.01636 0.01466 0.01331 0.01218 0.01112
    Cumulative Proportion  0.8863 0.90491 0.92128 0.93594 0.94925 0.96143 0.97255
                              PC29    PC30    PC31    PC32    PC33    PC34
    Standard deviation     0.57844 0.50440 0.39610 0.36041 0.28434 0.17755
    Proportion of Variance 0.00929 0.00707 0.00436 0.00361 0.00225 0.00088
    Cumulative Proportion  0.98184 0.98891 0.99327 0.99688 0.99912 1.00000
                                PC35      PC36
    Standard deviation     7.064e-15 4.305e-15
    Proportion of Variance 0.000e+00 0.000e+00
    Cumulative Proportion  1.000e+00 1.000e+00

    # Introduce new feature "Viral"
    News$Viral <- ifelse(News$shares > 1400, 1, 0)
    News$Viral <- as.factor(News$Viral)
    News.Scaled$shares <- News$shares
    News.Scaled$Viral <- ifelse(News.Scaled$shares > 1400, 1, 0)
    News.Scaled$Viral <- as.factor(News.Scaled$Viral)
    News.PCA$Viral <- ifelse(News.PCA$shares > 1400, 1, 0)
    News.PCA$Viral <- as.factor(News.PCA$Viral)

Methods: Baseline Performance Measure for Null Model
----------------------------------------------------

The baseline model accuracy we will use has an accuracy of about 50.66%,
which is derived by always assuming that the class belongs to the most
common class. In other words, if we assume every article is not viral,
we achieve an accuracy of 50.66%.

    yViral <- sum(News.PCA$Viral == 1)
    nViral <- sum(News.PCA$Viral == 0)
    BaselineAccuracy <- nViral/(yViral + nViral)
    print(BaselineAccuracy)
    [1] 0.5065584

Methods and Results: Implementing Regression Models for Subsequent Binary Classification
----------------------------------------------------------------------------------------

We implement three types of models: (1) linear regression, (2)
L2-regularized (ridge) linear regression, and (3) KNN. We implement
linear regression as a simple baseline performance comparison.
Meanwhile, we implement ridge regression to see if having implicit
feature select via sparsity through ridge regression helps improve
performance. Lastly, we use KNN as a non-parametric method to see if
this dataset performs better without underlying linearity assumptions.
Each model variant is run with a random 80/20 train/test split 20 times
to derive average performance. The following performance measurements
were calculated and printed as a table: accuracy, error, true positive
rate, false positive rate, true positives, false positives, true
negatives, and false negatives.

### Linear Regression and Subsequent Classification

Here we implement five variants of linear regression, described below:

-   Model 1: Linear regression on all features.
-   Model 2: Linear regression on top 5 principal components.
-   Model 3: Linear regression on top 10 principal components.
-   Model 4: Linear regression on top 15 principal components.
-   Model 5: Linear regression on top 20 principal components.

We see that Model 1, the linear regression model on all features without
PCA, performs the best in terms of an accuracy of about 50.12%. However,
we see that this model predicts nearly every article to be positive,
clearly indicating a somewhat useless model.

    # Find size of train and test sets for a 80/20 split
    n = nrow(News)
    n_train = round(0.8*n)  
    n_test = n - n_train

    # Examine linear regression performance on two datasets
    M <- 5 # number of models
    TP  <- matrix(0, nrow = 1, ncol = M)
    TN  <- matrix(0, nrow = 1, ncol = M)
    FP  <- matrix(0, nrow = 1, ncol = M)
    FN  <- matrix(0, nrow = 1, ncol = M)
    rmse_vals <- matrix(0, nrow = 20, ncol = M)

    for(i in 1:20){
      
      # Re-split into train and test sets
      train_cases = sample.int(n, n_train, replace=FALSE)
      test_cases = setdiff(1:n, train_cases)
      News_train = News.Scaled[train_cases,]
      News_test = News.Scaled[test_cases,]
      PNews_train = News.PCA[train_cases,]
      PNews_test = News.PCA[test_cases,]
      
      # LR model variants
      lm0 = lm(shares ~ . - Viral, data=News_train)
      lm1 = lm(shares ~ PC1 + PC2 + PC3 + PC4 + PC5, data=PNews_train)
      lm2 = lm(shares ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10, data=PNews_train)
      lm3 = lm(shares ~ - Viral - PC20 - PC19 - PC18 - PC17 - PC16, data=PNews_train)
      lm4 = lm(shares ~ - Viral, data=PNews_train)
      
      # Predictions out of sample
      yhat_test0 = predict(lm0, News_test)
      yhat_test1 = predict(lm1, PNews_test)
      yhat_test2 = predict(lm2, PNews_test)
      yhat_test3 = predict(lm3, PNews_test)
      yhat_test4 = predict(lm4, PNews_test)
      
      # Convert share predictions to factorized binary outcomes
      Binary0 <- ifelse(yhat_test0 <= 1400,0,1)
      Binary1 <- ifelse(yhat_test1 <= 1400,0,1)
      Binary2 <- ifelse(yhat_test2 <= 1400,0,1)
      Binary3 <- ifelse(yhat_test3 <= 1400,0,1)
      Binary4 <- ifelse(yhat_test4 <= 1400,0,1)
      
      Binary0 <- as.factor(Binary0)
      Binary1 <- as.factor(Binary1)
      Binary2 <- as.factor(Binary2)
      Binary3 <- as.factor(Binary3)
      Binary4 <- as.factor(Binary4)

      # Store RMSE values
      rmse_vals[i,1:M] <- c(rmse(News_test$shares,   yhat_test0),
                            rmse(PNews_test$shares,  yhat_test1),
                            rmse(PNews_test$shares,  yhat_test2),
                            rmse(PNews_test$shares,  yhat_test3),
                            rmse(PNews_test$shares,  yhat_test4))
      
      TP[1,1]  <- TP[1,1]  + sum(Binary0 == 1 & News_test$Viral == 1)
      TP[1,2]  <- TP[1,2]  + sum(Binary1 == 1 & PNews_test$Viral == 1)
      TP[1,3]  <- TP[1,3]  + sum(Binary2 == 1 & PNews_test$Viral == 1)
      TP[1,4]  <- TP[1,4]  + sum(Binary3 == 1 & PNews_test$Viral == 1)
      TP[1,5]  <- TP[1,5]  + sum(Binary4 == 1 & PNews_test$Viral == 1)
      
      TN[1,1]  <- TN[1,1]  + sum(Binary0 == 0 & News_test$Viral == 0)
      TN[1,2]  <- TN[1,2]  + sum(Binary1 == 0 & PNews_test$Viral == 0)
      TN[1,3]  <- TN[1,3]  + sum(Binary2 == 0 & PNews_test$Viral == 0)
      TN[1,4]  <- TN[1,4]  + sum(Binary3 == 0 & PNews_test$Viral == 0)
      TN[1,5]  <- TN[1,5]  + sum(Binary4 == 0 & PNews_test$Viral == 0)
      
      FP[1,1]  <- FP[1,1]  + sum(Binary0 == 1 & News_test$Viral == 0)
      FP[1,2]  <- FP[1,2]  + sum(Binary1 == 1 & PNews_test$Viral == 0)
      FP[1,3]  <- FP[1,3]  + sum(Binary2 == 1 & PNews_test$Viral == 0)
      FP[1,4]  <- FP[1,4]  + sum(Binary3 == 1 & PNews_test$Viral == 0)
      FP[1,5]  <- FP[1,5]  + sum(Binary4 == 1 & PNews_test$Viral == 0)
      
      FN[1,1]  <- FN[1,1]  + sum(Binary0 == 0 & News_test$Viral == 1)
      FN[1,2]  <- FN[1,2]  + sum(Binary1 == 0 & PNews_test$Viral == 1)
      FN[1,3]  <- FN[1,3]  + sum(Binary2 == 0 & PNews_test$Viral == 1)
      FN[1,4]  <- FN[1,4]  + sum(Binary3 == 0 & PNews_test$Viral == 1)
      FN[1,5]  <- FN[1,5]  + sum(Binary4 == 0 & PNews_test$Viral == 1)
    }

    # Print averaged results
    Results <- data.frame(as.vector((TP+TN)/(TP+TN+FP+FN)), as.vector((FP+FN)/(TP+TN+FP+FN)), as.vector(TP/(TP+FN)), as.vector(FP/(FP+TN)), 
                          as.vector(TP/20), as.vector(FP/20), as.vector(TN/20), as.vector(FN/20))
    names(Results) = c("Accuracy","Error","True Positive Rate","False Positive Rate","True Positive","False Positive","True Negative","False Negative")
    row.names(Results) = c("Model 1","Model 2","Model 3","Model 4","Model 5")
    kable(Results, caption = "Averaged performance of model variants.")

<table>
<caption>Averaged performance of model variants.</caption>
<thead>
<tr class="header">
<th></th>
<th align="right">Accuracy</th>
<th align="right">Error</th>
<th align="right">True Positive Rate</th>
<th align="right">False Positive Rate</th>
<th align="right">True Positive</th>
<th align="right">False Positive</th>
<th align="right">True Negative</th>
<th align="right">False Negative</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Model 1</td>
<td align="right">0.5012360</td>
<td align="right">0.4987640</td>
<td align="right">0.9914116</td>
<td align="right">0.9761872</td>
<td align="right">3878.65</td>
<td align="right">3921.10</td>
<td align="right">95.65</td>
<td align="right">33.60</td>
</tr>
<tr class="even">
<td>Model 2</td>
<td align="right">0.4934103</td>
<td align="right">0.5065897</td>
<td align="right">1.0000000</td>
<td align="right">1.0000000</td>
<td align="right">3912.25</td>
<td align="right">4016.75</td>
<td align="right">0.00</td>
<td align="right">0.00</td>
</tr>
<tr class="odd">
<td>Model 3</td>
<td align="right">0.4934166</td>
<td align="right">0.5065834</td>
<td align="right">0.9999872</td>
<td align="right">0.9999751</td>
<td align="right">3912.20</td>
<td align="right">4016.65</td>
<td align="right">0.10</td>
<td align="right">0.05</td>
</tr>
<tr class="even">
<td>Model 4</td>
<td align="right">0.4934103</td>
<td align="right">0.5065897</td>
<td align="right">1.0000000</td>
<td align="right">1.0000000</td>
<td align="right">3912.25</td>
<td align="right">4016.75</td>
<td align="right">0.00</td>
<td align="right">0.00</td>
</tr>
<tr class="odd">
<td>Model 5</td>
<td align="right">0.4934103</td>
<td align="right">0.5065897</td>
<td align="right">1.0000000</td>
<td align="right">1.0000000</td>
<td align="right">3912.25</td>
<td align="right">4016.75</td>
<td align="right">0.00</td>
<td align="right">0.00</td>
</tr>
</tbody>
</table>

### Ridge Regression and Subsequent Classification

Here we implement five variants of ridge regression, described below:

-   Model 1: Ridge regression on top 10 principal components with
    regularization strength *λ* = 0.01.
-   Model 2: Ridge regression on top 10 principal components with
    regularization strength *λ* = 0.1.
-   Model 3: Ridge regression on top 10 principal components with
    regularization strength *λ* = 1.
-   Model 4: Ridge regression on top 10 principal components with
    regularization strength *λ* = 10.
-   Model 5: Ridge regression on top 10 principal components with
    regularization strength *λ* = 100.

All model variants performed the same, with an accuracy of 49.31%. It is
also readily apparent that this model is faulty because it nearly
assumes every single article is positive, or viral.

    # Examine ridge regression performance
    M <- 5 # number of models
    TP  <- matrix(0, nrow = 1, ncol = M)
    TN  <- matrix(0, nrow = 1, ncol = M)
    FP  <- matrix(0, nrow = 1, ncol = M)
    FN  <- matrix(0, nrow = 1, ncol = M)
    rmse_vals <- matrix(0, nrow = 20, ncol = M)

    for(i in 1:20){
      
      # Re-split into train and test sets
      train_cases = sample.int(n, n_train, replace=FALSE)
      test_cases = setdiff(1:n, train_cases)
      News_train = News.Scaled[train_cases,]
      News_test = News.Scaled[test_cases,]
      PNews_train = News.PCA[train_cases,]
      PNews_test = News.PCA[test_cases,]
      
      # Fit ridge regression variants
      y <- PNews_train$shares
      x_train <- PNews_train %>% dplyr::select(PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9, PC10) %>% data.matrix()
      x_test  <- PNews_test  %>% dplyr::select(PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9, PC10) %>% data.matrix()
      fit0 <- glmnet(x_train, y, alpha = 0, lambda = 0.01)
      fit1 <- glmnet(x_train, y, alpha = 0, lambda = 0.1)
      fit2 <- glmnet(x_train, y, alpha = 0, lambda = 1)
      fit3 <- glmnet(x_train, y, alpha = 0, lambda = 10)
      fit4 <- glmnet(x_train, y, alpha = 0, lambda = 100)
      
      # Predict on test set
      Pred0 <- predict(fit0,x_test)
      Pred1 <- predict(fit1,x_test)
      Pred2 <- predict(fit2,x_test)
      Pred3 <- predict(fit3,x_test)
      Pred4 <- predict(fit4,x_test)
      
      
      # Convert share predictions to factorized binary outcomes
      Binary0 <- ifelse(Pred0 <= 1400,0,1)
      Binary1 <- ifelse(Pred1 <= 1400,0,1)
      Binary2 <- ifelse(Pred2 <= 1400,0,1)
      Binary3 <- ifelse(Pred3 <= 1400,0,1)
      Binary4 <- ifelse(Pred4 <= 1400,0,1)
      
      Binary0 <- as.factor(Binary0)
      Binary1 <- as.factor(Binary1)
      Binary2 <- as.factor(Binary2)
      Binary3 <- as.factor(Binary3)
      Binary4 <- as.factor(Binary4)

      # Store RMSE values
      rmse_vals[i,1:M] <- c(rmse(PNews_test$shares,  Pred0),
                            rmse(PNews_test$shares,  Pred1),
                            rmse(PNews_test$shares,  Pred2),
                            rmse(PNews_test$shares,  Pred3),
                            rmse(PNews_test$shares,  Pred4))
      
      TP[1,1]  <- TP[1,1]  + sum(Binary0 == 1 & PNews_test$Viral == 1)
      TP[1,2]  <- TP[1,2]  + sum(Binary1 == 1 & PNews_test$Viral == 1)
      TP[1,3]  <- TP[1,3]  + sum(Binary2 == 1 & PNews_test$Viral == 1)
      TP[1,4]  <- TP[1,4]  + sum(Binary3 == 1 & PNews_test$Viral == 1)
      TP[1,5]  <- TP[1,5]  + sum(Binary4 == 1 & PNews_test$Viral == 1)
      
      TN[1,1]  <- TN[1,1]  + sum(Binary0 == 0 & PNews_test$Viral == 0)
      TN[1,2]  <- TN[1,2]  + sum(Binary1 == 0 & PNews_test$Viral == 0)
      TN[1,3]  <- TN[1,3]  + sum(Binary2 == 0 & PNews_test$Viral == 0)
      TN[1,4]  <- TN[1,4]  + sum(Binary3 == 0 & PNews_test$Viral == 0)
      TN[1,5]  <- TN[1,5]  + sum(Binary4 == 0 & PNews_test$Viral == 0)
      
      FP[1,1]  <- FP[1,1]  + sum(Binary0 == 1 & PNews_test$Viral == 0)
      FP[1,2]  <- FP[1,2]  + sum(Binary1 == 1 & PNews_test$Viral == 0)
      FP[1,3]  <- FP[1,3]  + sum(Binary2 == 1 & PNews_test$Viral == 0)
      FP[1,4]  <- FP[1,4]  + sum(Binary3 == 1 & PNews_test$Viral == 0)
      FP[1,5]  <- FP[1,5]  + sum(Binary4 == 1 & PNews_test$Viral == 0)
      
      FN[1,1]  <- FN[1,1]  + sum(Binary0 == 0 & PNews_test$Viral == 1)
      FN[1,2]  <- FN[1,2]  + sum(Binary1 == 0 & PNews_test$Viral == 1)
      FN[1,3]  <- FN[1,3]  + sum(Binary2 == 0 & PNews_test$Viral == 1)
      FN[1,4]  <- FN[1,4]  + sum(Binary3 == 0 & PNews_test$Viral == 1)
      FN[1,5]  <- FN[1,5]  + sum(Binary4 == 0 & PNews_test$Viral == 1)
    }

    # Print averaged results
    Results <- data.frame(as.vector((TP+TN)/(TP+TN+FP+FN)), as.vector((FP+FN)/(TP+TN+FP+FN)), as.vector(TP/(TP+FN)), as.vector(FP/(FP+TN)), 
                          as.vector(TP/20), as.vector(FP/20), as.vector(TN/20), as.vector(FN/20))
    names(Results) = c("Accuracy","Error","True Positive Rate","False Positive Rate","True Positive","False Positive","True Negative","False Negative")
    row.names(Results) = c("Model 1","Model 2","Model 3","Model 4","Model 5")
    kable(Results, caption = "Averaged performance of model variants.")

<table>
<caption>Averaged performance of model variants.</caption>
<thead>
<tr class="header">
<th></th>
<th align="right">Accuracy</th>
<th align="right">Error</th>
<th align="right">True Positive Rate</th>
<th align="right">False Positive Rate</th>
<th align="right">True Positive</th>
<th align="right">False Positive</th>
<th align="right">True Negative</th>
<th align="right">False Negative</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Model 1</td>
<td align="right">0.4930760</td>
<td align="right">0.5069240</td>
<td align="right">0.9999872</td>
<td align="right">0.9999627</td>
<td align="right">3909.45</td>
<td align="right">4019.35</td>
<td align="right">0.15</td>
<td align="right">0.05</td>
</tr>
<tr class="even">
<td>Model 2</td>
<td align="right">0.4930760</td>
<td align="right">0.5069240</td>
<td align="right">0.9999872</td>
<td align="right">0.9999627</td>
<td align="right">3909.45</td>
<td align="right">4019.35</td>
<td align="right">0.15</td>
<td align="right">0.05</td>
</tr>
<tr class="odd">
<td>Model 3</td>
<td align="right">0.4930760</td>
<td align="right">0.5069240</td>
<td align="right">0.9999872</td>
<td align="right">0.9999627</td>
<td align="right">3909.45</td>
<td align="right">4019.35</td>
<td align="right">0.15</td>
<td align="right">0.05</td>
</tr>
<tr class="even">
<td>Model 4</td>
<td align="right">0.4930697</td>
<td align="right">0.5069303</td>
<td align="right">0.9999872</td>
<td align="right">0.9999751</td>
<td align="right">3909.45</td>
<td align="right">4019.40</td>
<td align="right">0.10</td>
<td align="right">0.05</td>
</tr>
<tr class="odd">
<td>Model 5</td>
<td align="right">0.4930760</td>
<td align="right">0.5069240</td>
<td align="right">1.0000000</td>
<td align="right">0.9999751</td>
<td align="right">3909.50</td>
<td align="right">4019.40</td>
<td align="right">0.10</td>
<td align="right">0.00</td>
</tr>
</tbody>
</table>

### KNN Regression and Subsequent Classification

Here we implement five variants of KNN regression, described below:

-   Model 1: KNN on top 10 principal components with *k* = 1.
-   Model 2: KNN on top 10 principal components with *k* = 3.
-   Model 3: KNN on top 10 principal components with *k* = 5.
-   Model 4: KNN on top 10 principal components with *k* = 7.
-   Model 5: KNN on top 10 principal components with *k* = 10.

We see that out of this group, the best performing model in terms of
overall accuracy is the KNN regression model with *k* = 3, giving an
accuracy of about 55.83%.

    # Examine KNN regression performance
    M <- 5 # number of models
    TP  <- matrix(0, nrow = 1, ncol = M)
    TN  <- matrix(0, nrow = 1, ncol = M)
    FP  <- matrix(0, nrow = 1, ncol = M)
    FN  <- matrix(0, nrow = 1, ncol = M)
    rmse_vals <- matrix(0, nrow = 20, ncol = M)

    for(i in 1:20){
      
      # Re-split into train and test sets
      train_cases = sample.int(n, n_train, replace=FALSE)
      test_cases = setdiff(1:n, train_cases)
      PNews_train = News.PCA[train_cases,]
      PNews_test = News.PCA[test_cases,]
      
      # KNN model variants
      KNN01 <- knn.reg(train = PNews_train[c(1:10)], test = PNews_test[c(1:10)], y = PNews_train[c(21)], k = 1)
      KNN03 <- knn.reg(train = PNews_train[c(1:10)], test = PNews_test[c(1:10)], y = PNews_train[c(21)], k = 3)
      KNN05 <- knn.reg(train = PNews_train[c(1:10)], test = PNews_test[c(1:10)], y = PNews_train[c(21)], k = 5)
      KNN07 <- knn.reg(train = PNews_train[c(1:10)], test = PNews_test[c(1:10)], y = PNews_train[c(21)], k = 7)
      KNN09 <- knn.reg(train = PNews_train[c(1:10)], test = PNews_test[c(1:10)], y = PNews_train[c(21)], k = 10)
      
      # Convert share predictions to factorized binary outcomes
      Binary0 <- ifelse(KNN01$pred <= 1400,0,1)
      Binary1 <- ifelse(KNN03$pred <= 1400,0,1)
      Binary2 <- ifelse(KNN05$pred <= 1400,0,1)
      Binary3 <- ifelse(KNN07$pred <= 1400,0,1)
      Binary4 <- ifelse(KNN09$pred <= 1400,0,1)
      
      Binary0 <- as.factor(Binary0)
      Binary1 <- as.factor(Binary1)
      Binary2 <- as.factor(Binary2)
      Binary3 <- as.factor(Binary3)
      Binary4 <- as.factor(Binary4)

      # Store RMSE values
      rmse_vals[i,1:M] <- c(rmse(PNews_test$shares,  KNN01$pred),
                            rmse(PNews_test$shares,  KNN03$pred),
                            rmse(PNews_test$shares,  KNN05$pred),
                            rmse(PNews_test$shares,  KNN07$pred),
                            rmse(PNews_test$shares,  KNN09$pred))
      
      TP[1,1]  <- TP[1,1]  + sum(Binary0 == 1 & PNews_test$Viral == 1)
      TP[1,2]  <- TP[1,2]  + sum(Binary1 == 1 & PNews_test$Viral == 1)
      TP[1,3]  <- TP[1,3]  + sum(Binary2 == 1 & PNews_test$Viral == 1)
      TP[1,4]  <- TP[1,4]  + sum(Binary3 == 1 & PNews_test$Viral == 1)
      TP[1,5]  <- TP[1,5]  + sum(Binary4 == 1 & PNews_test$Viral == 1)
      
      TN[1,1]  <- TN[1,1]  + sum(Binary0 == 0 & PNews_test$Viral == 0)
      TN[1,2]  <- TN[1,2]  + sum(Binary1 == 0 & PNews_test$Viral == 0)
      TN[1,3]  <- TN[1,3]  + sum(Binary2 == 0 & PNews_test$Viral == 0)
      TN[1,4]  <- TN[1,4]  + sum(Binary3 == 0 & PNews_test$Viral == 0)
      TN[1,5]  <- TN[1,5]  + sum(Binary4 == 0 & PNews_test$Viral == 0)
      
      FP[1,1]  <- FP[1,1]  + sum(Binary0 == 1 & PNews_test$Viral == 0)
      FP[1,2]  <- FP[1,2]  + sum(Binary1 == 1 & PNews_test$Viral == 0)
      FP[1,3]  <- FP[1,3]  + sum(Binary2 == 1 & PNews_test$Viral == 0)
      FP[1,4]  <- FP[1,4]  + sum(Binary3 == 1 & PNews_test$Viral == 0)
      FP[1,5]  <- FP[1,5]  + sum(Binary4 == 1 & PNews_test$Viral == 0)
      
      FN[1,1]  <- FN[1,1]  + sum(Binary0 == 0 & PNews_test$Viral == 1)
      FN[1,2]  <- FN[1,2]  + sum(Binary1 == 0 & PNews_test$Viral == 1)
      FN[1,3]  <- FN[1,3]  + sum(Binary2 == 0 & PNews_test$Viral == 1)
      FN[1,4]  <- FN[1,4]  + sum(Binary3 == 0 & PNews_test$Viral == 1)
      FN[1,5]  <- FN[1,5]  + sum(Binary4 == 0 & PNews_test$Viral == 1)
    }

    # Print averaged results
    Results <- data.frame(as.vector((TP+TN)/(TP+TN+FP+FN)), as.vector((FP+FN)/(TP+TN+FP+FN)), as.vector(TP/(TP+FN)), as.vector(FP/(FP+TN)), 
                          as.vector(TP/20), as.vector(FP/20), as.vector(TN/20), as.vector(FN/20))
    names(Results) = c("Accuracy","Error","True Positive Rate","False Positive Rate","True Positive","False Positive","True Negative","False Negative")
    row.names(Results) = c("Model 1","Model 2","Model 3","Model 4","Model 5")
    kable(Results, caption = "Averaged performance of model variants.")

<table>
<caption>Averaged performance of model variants.</caption>
<thead>
<tr class="header">
<th></th>
<th align="right">Accuracy</th>
<th align="right">Error</th>
<th align="right">True Positive Rate</th>
<th align="right">False Positive Rate</th>
<th align="right">True Positive</th>
<th align="right">False Positive</th>
<th align="right">True Negative</th>
<th align="right">False Negative</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Model 1</td>
<td align="right">0.5553790</td>
<td align="right">0.4446210</td>
<td align="right">0.5395364</td>
<td align="right">0.4292007</td>
<td align="right">2110.10</td>
<td align="right">1724.55</td>
<td align="right">2293.50</td>
<td align="right">1800.85</td>
</tr>
<tr class="even">
<td>Model 2</td>
<td align="right">0.5583491</td>
<td align="right">0.4416509</td>
<td align="right">0.7344251</td>
<td align="right">0.6130337</td>
<td align="right">2872.30</td>
<td align="right">2463.20</td>
<td align="right">1554.85</td>
<td align="right">1038.65</td>
</tr>
<tr class="odd">
<td>Model 3</td>
<td align="right">0.5559844</td>
<td align="right">0.4440156</td>
<td align="right">0.8123090</td>
<td align="right">0.6935080</td>
<td align="right">3176.90</td>
<td align="right">2786.55</td>
<td align="right">1231.50</td>
<td align="right">734.05</td>
</tr>
<tr class="even">
<td>Model 4</td>
<td align="right">0.5530521</td>
<td align="right">0.4469479</td>
<td align="right">0.8560324</td>
<td align="right">0.7418524</td>
<td align="right">3347.90</td>
<td align="right">2980.80</td>
<td align="right">1037.25</td>
<td align="right">563.05</td>
</tr>
<tr class="odd">
<td>Model 5</td>
<td align="right">0.5449552</td>
<td align="right">0.4550448</td>
<td align="right">0.8923791</td>
<td align="right">0.7932081</td>
<td align="right">3490.05</td>
<td align="right">3187.15</td>
<td align="right">830.90</td>
<td align="right">420.90</td>
</tr>
</tbody>
</table>

Methods and Results: Implementing Classification Models to Predict Viral Status
-------------------------------------------------------------------------------

We implement two types of classification models: (1) logistic regression
and (2) KNN. We implement simple logistic regression as a simple
baseline performance comparison. Lastly, we use KNN as a non-parametric
method to see if this dataset performs better without underlying
linearity assumptions. Each model variant is run with a random 80/20
train/test split 20 times to derive average performance.

### Logistic Regression for Classification

Here we implement five variants of logistic regression, described below:

-   Model 1: Logistic regression on all features.
-   Model 2: Logistic regression on top 5 principal components.
-   Model 3: Logistic regression on top 10 principal components.
-   Model 4: Logistic regression on top 15 principal components.
-   Model 5: Logistic regression on top 20 principal components.

We see that out of this group, the best performing model in terms of
overall accuracy is the logistic model with all features without PCA,
giving an accuracy of about 59.12%. It is worth mentioning that this
performance is higher than both linear regression and ridge regression
models used in the regression-first models in previous sections. The
logistic regression model also moderately outperforms the KNN regression
model.

    # Examine logistic regression performance
    M <- 5 # number of models
    TP  <- matrix(0, nrow = 1, ncol = M)
    TN  <- matrix(0, nrow = 1, ncol = M)
    FP  <- matrix(0, nrow = 1, ncol = M)
    FN  <- matrix(0, nrow = 1, ncol = M)

    for(i in 1:20){
      
      # Re-split into train and test sets
      train_cases = sample.int(n, n_train, replace=FALSE)
      test_cases = setdiff(1:n, train_cases)
      News_train = News.Scaled[train_cases,]
      News_test = News.Scaled[test_cases,]
      PNews_train = News.PCA[train_cases,]
      PNews_test = News.PCA[test_cases,]
      
      # Fit logistic regression variants
      fit0 <- glm(Viral ~ . - shares, data = News_train, family = "binomial")
      fit1 <- glm(Viral ~ PC1 + PC2 + PC3 + PC4 + PC5, data = PNews_train, family = "binomial")
      fit2 <- glm(Viral ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10, data = PNews_train, family = "binomial")
      fit3 <- glm(Viral ~ . - shares - PC20 - PC19 - PC18 - PC17 - PC16, data = PNews_train, family = "binomial")
      fit4 <- glm(Viral ~ . - shares, data = PNews_train, family = "binomial")
      
      # Predict on test set
      Pred0 <- predict(fit0,News_test)
      Pred1 <- predict(fit1,PNews_test)
      Pred2 <- predict(fit2,PNews_test)
      Pred3 <- predict(fit3,PNews_test)
      Pred4 <- predict(fit4,PNews_test)
      
      # Convert share predictions to factorized binary outcomes
      Binary0 <- ifelse(Pred0 <= 0.5,0,1)
      Binary1 <- ifelse(Pred1 <= 0.5,0,1)
      Binary2 <- ifelse(Pred2 <= 0.5,0,1)
      Binary3 <- ifelse(Pred3 <= 0.5,0,1)
      Binary4 <- ifelse(Pred4 <= 0.5,0,1)
      
      TP[1,1]  <- TP[1,1]  + sum(Binary0 == 1 & News_test$Viral == 1)
      TP[1,2]  <- TP[1,2]  + sum(Binary1 == 1 & PNews_test$Viral == 1)
      TP[1,3]  <- TP[1,3]  + sum(Binary2 == 1 & PNews_test$Viral == 1)
      TP[1,4]  <- TP[1,4]  + sum(Binary3 == 1 & PNews_test$Viral == 1)
      TP[1,5]  <- TP[1,5]  + sum(Binary4 == 1 & PNews_test$Viral == 1)
      
      TN[1,1]  <- TN[1,1]  + sum(Binary0 == 0 & News_test$Viral == 0)
      TN[1,2]  <- TN[1,2]  + sum(Binary1 == 0 & PNews_test$Viral == 0)
      TN[1,3]  <- TN[1,3]  + sum(Binary2 == 0 & PNews_test$Viral == 0)
      TN[1,4]  <- TN[1,4]  + sum(Binary3 == 0 & PNews_test$Viral == 0)
      TN[1,5]  <- TN[1,5]  + sum(Binary4 == 0 & PNews_test$Viral == 0)
      
      FP[1,1]  <- FP[1,1]  + sum(Binary0 == 1 & News_test$Viral == 0)
      FP[1,2]  <- FP[1,2]  + sum(Binary1 == 1 & PNews_test$Viral == 0)
      FP[1,3]  <- FP[1,3]  + sum(Binary2 == 1 & PNews_test$Viral == 0)
      FP[1,4]  <- FP[1,4]  + sum(Binary3 == 1 & PNews_test$Viral == 0)
      FP[1,5]  <- FP[1,5]  + sum(Binary4 == 1 & PNews_test$Viral == 0)

      FN[1,1]  <- FN[1,1]  + sum(Binary0 == 0 & News_test$Viral == 1)
      FN[1,2]  <- FN[1,2]  + sum(Binary1 == 0 & PNews_test$Viral == 1)
      FN[1,3]  <- FN[1,3]  + sum(Binary2 == 0 & PNews_test$Viral == 1)
      FN[1,4]  <- FN[1,4]  + sum(Binary3 == 0 & PNews_test$Viral == 1)
      FN[1,5]  <- FN[1,5]  + sum(Binary4 == 0 & PNews_test$Viral == 1)
    }

    # Print averaged results
    Results <- data.frame(as.vector((TP+TN)/(TP+TN+FP+FN)), as.vector((FP+FN)/(TP+TN+FP+FN)), as.vector(TP/(TP+FN)),        
                          as.vector(FP/(FP+TN)), as.vector(TP/20), as.vector(FP/20), as.vector(TN/20), as.vector(FN/20))
    names(Results) = c("Accuracy","Error","True Positive Rate","False Positive Rate","True Positive","False Positive","True Negative","False Negative")
    row.names(Results) = c("Model 1","Model 2","Model 3","Model 4","Model 5")
    kable(Results, caption = "Averaged performance of model variants.")

<table>
<caption>Averaged performance of model variants.</caption>
<thead>
<tr class="header">
<th></th>
<th align="right">Accuracy</th>
<th align="right">Error</th>
<th align="right">True Positive Rate</th>
<th align="right">False Positive Rate</th>
<th align="right">True Positive</th>
<th align="right">False Positive</th>
<th align="right">True Negative</th>
<th align="right">False Negative</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Model 1</td>
<td align="right">0.5916761</td>
<td align="right">0.4083239</td>
<td align="right">0.2808245</td>
<td align="right">0.1040356</td>
<td align="right">1101.45</td>
<td align="right">416.85</td>
<td align="right">3589.95</td>
<td align="right">2820.75</td>
</tr>
<tr class="even">
<td>Model 2</td>
<td align="right">0.5571194</td>
<td align="right">0.4428806</td>
<td align="right">0.1832645</td>
<td align="right">0.0769192</td>
<td align="right">718.80</td>
<td align="right">308.20</td>
<td align="right">3698.60</td>
<td align="right">3203.40</td>
</tr>
<tr class="odd">
<td>Model 3</td>
<td align="right">0.5643965</td>
<td align="right">0.4356035</td>
<td align="right">0.1989572</td>
<td align="right">0.0778801</td>
<td align="right">780.35</td>
<td align="right">312.05</td>
<td align="right">3694.75</td>
<td align="right">3141.85</td>
</tr>
<tr class="even">
<td>Model 4</td>
<td align="right">0.5685774</td>
<td align="right">0.4314226</td>
<td align="right">0.2160777</td>
<td align="right">0.0863657</td>
<td align="right">847.50</td>
<td align="right">346.05</td>
<td align="right">3660.75</td>
<td align="right">3074.70</td>
</tr>
<tr class="odd">
<td>Model 5</td>
<td align="right">0.5766364</td>
<td align="right">0.4233636</td>
<td align="right">0.2406940</td>
<td align="right">0.0945143</td>
<td align="right">944.05</td>
<td align="right">378.70</td>
<td align="right">3628.10</td>
<td align="right">2978.15</td>
</tr>
</tbody>
</table>

### KNN for Classification

Here we implement six variants of KNN classification, described below:

-   Model 1: KNN on top 10 principal components with *k* = 1.
-   Model 2: KNN on top 10 principal components with *k* = 3.
-   Model 3: KNN on top 10 principal components with *k* = 5.
-   Model 4: KNN on top 10 principal components with *k* = 10.
-   Model 5: KNN on top 10 principal components with *k* = 15.
-   Model 6: KNN on top 10 principal components with *k* = 20.

We see that out of this group, the best performing model in terms of
overall accuracy is the KNN model with *k* = 20, giving an accuracy of
about 60.95%.

    # Examine KNN classification performance
    M <- 6 # number of models
    TP  <- matrix(0, nrow = 1, ncol = M)
    TN  <- matrix(0, nrow = 1, ncol = M)
    FP  <- matrix(0, nrow = 1, ncol = M)
    FN  <- matrix(0, nrow = 1, ncol = M)

    for(i in 1:20){
      
      # Re-split into train and test sets
      train_cases = sample.int(n, n_train, replace=FALSE)
      test_cases = setdiff(1:n, train_cases)
      PNews_train = News.PCA[train_cases,]
      PNews_test = News.PCA[test_cases,]
      
      # Fit KNN variants and generate predictions
      Pred0 <- knn(train = PNews_train[c(1:10)], test = PNews_test[c(1:10)], cl = PNews_train$Viral, k = 1)
      Pred1 <- knn(train = PNews_train[c(1:10)], test = PNews_test[c(1:10)], cl = PNews_train$Viral, k = 3)
      Pred2 <- knn(train = PNews_train[c(1:10)], test = PNews_test[c(1:10)], cl = PNews_train$Viral, k = 5)
      Pred3 <- knn(train = PNews_train[c(1:10)], test = PNews_test[c(1:10)], cl = PNews_train$Viral, k = 10)
      Pred4 <- knn(train = PNews_train[c(1:10)], test = PNews_test[c(1:10)], cl = PNews_train$Viral, k = 15)
      Pred5 <- knn(train = PNews_train[c(1:10)], test = PNews_test[c(1:10)], cl = PNews_train$Viral, k = 20)
      
      TP[1,1]  <- TP[1,1]  + sum(Pred0 == 1 & PNews_test$Viral == 1)
      TP[1,2]  <- TP[1,2]  + sum(Pred1 == 1 & PNews_test$Viral == 1)
      TP[1,3]  <- TP[1,3]  + sum(Pred2 == 1 & PNews_test$Viral == 1)
      TP[1,4]  <- TP[1,4]  + sum(Pred3 == 1 & PNews_test$Viral == 1)
      TP[1,5]  <- TP[1,5]  + sum(Pred4 == 1 & PNews_test$Viral == 1)
      TP[1,6]  <- TP[1,6]  + sum(Pred5 == 1 & PNews_test$Viral == 1)
      
      TN[1,1]  <- TN[1,1]  + sum(Pred0 == 0 & PNews_test$Viral == 0)
      TN[1,2]  <- TN[1,2]  + sum(Pred1 == 0 & PNews_test$Viral == 0)
      TN[1,3]  <- TN[1,3]  + sum(Pred2 == 0 & PNews_test$Viral == 0)
      TN[1,4]  <- TN[1,4]  + sum(Pred3 == 0 & PNews_test$Viral == 0)
      TN[1,5]  <- TN[1,5]  + sum(Pred4 == 0 & PNews_test$Viral == 0)
      TN[1,6]  <- TN[1,6]  + sum(Pred5 == 0 & PNews_test$Viral == 0)
      
      FP[1,1]  <- FP[1,1]  + sum(Pred0 == 1 & PNews_test$Viral == 0)
      FP[1,2]  <- FP[1,2]  + sum(Pred1 == 1 & PNews_test$Viral == 0)
      FP[1,3]  <- FP[1,3]  + sum(Pred2 == 1 & PNews_test$Viral == 0)
      FP[1,4]  <- FP[1,4]  + sum(Pred3 == 1 & PNews_test$Viral == 0)
      FP[1,5]  <- FP[1,5]  + sum(Pred4 == 1 & PNews_test$Viral == 0)
      FP[1,6]  <- FP[1,6]  + sum(Pred5 == 1 & PNews_test$Viral == 0)

      FN[1,1]  <- FN[1,1]  + sum(Pred0 == 0 & PNews_test$Viral == 1)
      FN[1,2]  <- FN[1,2]  + sum(Pred1 == 0 & PNews_test$Viral == 1)
      FN[1,3]  <- FN[1,3]  + sum(Pred2 == 0 & PNews_test$Viral == 1)
      FN[1,4]  <- FN[1,4]  + sum(Pred3 == 0 & PNews_test$Viral == 1)
      FN[1,5]  <- FN[1,5]  + sum(Pred4 == 0 & PNews_test$Viral == 1)
      FN[1,6]  <- FN[1,6]  + sum(Pred5 == 0 & PNews_test$Viral == 1)
    }

    # Print averaged results
    Results <- data.frame(as.vector((TP+TN)/(TP+TN+FP+FN)), as.vector((FP+FN)/(TP+TN+FP+FN)), as.vector(TP/(TP+FN)),
                          as.vector(FP/(FP+TN)), as.vector(TP/20), as.vector(FP/20), as.vector(TN/20), as.vector(FN/20))
    names(Results) = c("Accuracy","Error","True Positive Rate","False Positive Rate","True Positive","False Positive","True Negative","False Negative")
    row.names(Results) = c("Model 1","Model 2","Model 3","Model 4","Model 5","Model 6")
    kable(Results, caption = "Averaged performance of model variants.")

<table>
<caption>Averaged performance of model variants.</caption>
<thead>
<tr class="header">
<th></th>
<th align="right">Accuracy</th>
<th align="right">Error</th>
<th align="right">True Positive Rate</th>
<th align="right">False Positive Rate</th>
<th align="right">True Positive</th>
<th align="right">False Positive</th>
<th align="right">True Negative</th>
<th align="right">False Negative</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Model 1</td>
<td align="right">0.5550574</td>
<td align="right">0.4449426</td>
<td align="right">0.5401300</td>
<td align="right">0.4303208</td>
<td align="right">2119.20</td>
<td align="right">1723.65</td>
<td align="right">2281.85</td>
<td align="right">1804.30</td>
</tr>
<tr class="even">
<td>Model 2</td>
<td align="right">0.5736789</td>
<td align="right">0.4263211</td>
<td align="right">0.5460176</td>
<td align="right">0.3992261</td>
<td align="right">2142.30</td>
<td align="right">1599.10</td>
<td align="right">2406.40</td>
<td align="right">1781.20</td>
</tr>
<tr class="odd">
<td>Model 3</td>
<td align="right">0.5845315</td>
<td align="right">0.4154685</td>
<td align="right">0.5529374</td>
<td align="right">0.3845213</td>
<td align="right">2169.45</td>
<td align="right">1540.20</td>
<td align="right">2465.30</td>
<td align="right">1754.05</td>
</tr>
<tr class="even">
<td>Model 4</td>
<td align="right">0.5952075</td>
<td align="right">0.4047925</td>
<td align="right">0.4733656</td>
<td align="right">0.2854450</td>
<td align="right">1857.25</td>
<td align="right">1143.35</td>
<td align="right">2862.15</td>
<td align="right">2066.25</td>
</tr>
<tr class="odd">
<td>Model 5</td>
<td align="right">0.6067789</td>
<td align="right">0.3932211</td>
<td align="right">0.5633108</td>
<td align="right">0.3506429</td>
<td align="right">2210.15</td>
<td align="right">1404.50</td>
<td align="right">2601.00</td>
<td align="right">1713.35</td>
</tr>
<tr class="even">
<td>Model 6</td>
<td align="right">0.6095094</td>
<td align="right">0.3904906</td>
<td align="right">0.5172168</td>
<td align="right">0.3000874</td>
<td align="right">2029.30</td>
<td align="right">1202.00</td>
<td align="right">2803.50</td>
<td align="right">1894.20</td>
</tr>
</tbody>
</table>

Conclusion: Classification-First Models Perform Better Than Regression-First Models
-----------------------------------------------------------------------------------

In terms of accuracy, our best regression-first model was a KNN model
with *k* = 3, achieving an accuracy of about 55.99%. Meanwhile, our best
classification-first model was a KNN model with *k* = 120, achieving an
accuracy of about 60.95%. Consequently, it becomes suggestive that
non-parametric models may perform better than parametric models in both
regression-first and classification-first scenarios. Perhaps more
accurately stated, it appears that the dataset cannot be effectively
classified through linear methods such as linear regression, ridge
regression, or logistic regression. Consequently, nonlinear methods such
as KNN prevails.

Lastly, despite the somewhat crude and unthorough experiments, there is
overwhelming evidence that a classification-first approach yields much
better performance. This is most likely due to the fact that from a
binary classification standpoint, we have a nearly perfectly balanced
dataset with 19562 viral articles and 20082 nonviral articles. In
contrast, we can easily see that there is an uneven distribution of
articles in terms of number of shares. In regression methods, unbalanced
classes (or in the case of continuous values, unbalanced distributions)
can result in biased (in the loosest definition of bias) models because
points within certain regions have a heavier influence on the ultimate
regression model. Meanwhile, in KNN models, unbalanced classes have a
direct impact on predictions since the model relies on similar points in
the feature space to make its output prediction. If one class heavily
outweights the other, the predominant class would be more likely to be
predicted in a KNN model.

From these experiments alone, the extent of unbalanced distributions in
the feature "shares" interfering with the perforamnce of the model is
unclear and unproven. Perhaps by resampling the dataset to achieve a
homogenous distribution in number of shares in each article and
examining the performances of the models thereafter, would provide
insight on whether this distribution imbalance is truly causing the
regression-first approach to underperform.
