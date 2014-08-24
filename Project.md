Project for determing the class of a given exercise 
========================================================
Outline - I am downloading the data for the training set and I am partitioning it into two parts of sizes 75% and 25%. The former will be my training set and the later will be my testing set within the larger framework of the training set I will test my model on the testing part and check the quality of the predictions and once satisfied, it will be applied to the **testing set provided** to us. 


```r
#Downloading data 
fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(fileUrl, destfile="pml-training.csv", method="curl")
FitbitTrain<-read.csv("pml-training.csv")
```
We are creating the test and train sets

```r
library(caret);library(psych)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
## 
## Attaching package: 'psych'
## 
## The following object is masked from 'package:ggplot2':
## 
##     %+%
```

```r
#important- here I am using stringsAsFactors=FALSE, so it will convert the classe variable into a character variable and not leave it as a factor variable as it is).
#Here an indexing set it being created using classe which will be the object we 
#want to predict later on. 
trainIndex <- createDataPartition(FitbitTrain$classe,p=0.75,list=FALSE)
traininginFitbitTrain<-FitbitTrain[trainIndex,]
testinginFitbitTrain<- FitbitTrain[-trainIndex,]
##dim(FitbitTrain); dim(trainIndex); were checked to be correct.
###Now I am trying to reduce the variables that I need to fit the data
##Step 1 is to determine the onces which enteries are characters,looking that data
## shows that the enteries which are characters, seem to be derived from the data or is 
##information like name, time, date, etc. 
#This puts the list of all such columns together
class <- c()
for (i in 1:160){
      if (class(FitbitTrain[,i])=="factor")
      {class <- c(class,i)}
}
class
```

```
##  [1]   2   5   6  12  13  14  15  16  17  20  23  26  69  70  71  72  73
## [18]  74  87  88  89  90  91  92  95  98 101 125 126 127 128 129 130 133
## [35] 136 139 160
```

```r
##Now we remove variables with very small variance, for this we list those variables
zerovar <- nearZeroVar(FitbitTrain)
## Now we remove the enteries which are sparse. For this we use the describe the function 
#which is available using the psych package.The second entry gives the number of row enteries. 
#These numbers vary between 19622 and 4** 
sparse <- which(describe(FitbitTrain)[,2]<19622) 
invalid <- append(append(sparse,zerovar),class)
invalid <- unique(sort(invalid))
##Here is put together all the features that I would like to ignore.  
invalid
```

```
##   [1]   2   5   6  12  13  14  15  16  17  18  19  20  21  22  23  24  25
##  [18]  26  27  28  29  30  31  32  33  34  35  36  50  51  52  53  54  55
##  [35]  56  57  58  59  69  70  71  72  73  74  75  76  77  78  79  80  81
##  [52]  82  83  87  88  89  90  91  92  93  94  95  96  97  98  99 100 101
##  [69] 103 104 105 106 107 108 109 110 111 112 125 126 127 128 129 130 131
##  [86] 132 133 134 135 136 137 138 139 141 142 143 144 145 146 147 148 149
## [103] 150 160
```

```r
#invalid <- invalid[-length(invalid)]
#trainIndex <- createDataPartition(FitbitTrain$classe,p=0.75,list=FALSE)
traininginFitbitTrain1<-FitbitTrain[trainIndex,-invalid]
#str(traininginFitbitTrain1)
testinginFitbitTrain1<- FitbitTrain[-trainIndex,-invalid]
#str(testinginFitbitTrain1)

##dim(FitbitTrain); dim(trainIndex); were checked to be correct.
invalid1 <- invalid[1:length(invalid)-1]
invalid1
```

```
##   [1]   2   5   6  12  13  14  15  16  17  18  19  20  21  22  23  24  25
##  [18]  26  27  28  29  30  31  32  33  34  35  36  50  51  52  53  54  55
##  [35]  56  57  58  59  69  70  71  72  73  74  75  76  77  78  79  80  81
##  [52]  82  83  87  88  89  90  91  92  93  94  95  96  97  98  99 100 101
##  [69] 103 104 105 106 107 108 109 110 111 112 125 126 127 128 129 130 131
##  [86] 132 133 134 135 136 137 138 139 141 142 143 144 145 146 147 148 149
## [103] 150
```

```r
##I want to ensure that I put back the classe feature into my data:

traininginFitbitTrain2<-FitbitTrain[trainIndex,-invalid1]
testinginFitbitTrain2<-FitbitTrain[-trainIndex,-invalid1]


###Now I am doing the training and the train data


preProc <- preProcess(traininginFitbitTrain1, method="pca")
#str(traininginFitbitTrain1)
valuesintraininginFitbitTrain1 <-  predict(preProc,traininginFitbitTrain1)
#head(valuesintraininginFitbitTrain2)
#head(traininginFitbitTrain2)

modelintraininginFitbitTrain2 <- train(traininginFitbitTrain2$classe ~., data=valuesintraininginFitbitTrain1, method="rf", trControl = trainControl(method = "cv", number = 4, allowParallel = TRUE,verboseIter = TRUE))
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## 
## The following object is masked from 'package:psych':
## 
##     outlier
```

```
## + Fold1: mtry= 2 
## - Fold1: mtry= 2 
## + Fold1: mtry=15 
## - Fold1: mtry=15 
## + Fold1: mtry=29 
## - Fold1: mtry=29 
## + Fold2: mtry= 2 
## - Fold2: mtry= 2 
## + Fold2: mtry=15 
## - Fold2: mtry=15 
## + Fold2: mtry=29 
## - Fold2: mtry=29 
## + Fold3: mtry= 2 
## - Fold3: mtry= 2 
## + Fold3: mtry=15 
## - Fold3: mtry=15 
## + Fold3: mtry=29 
## - Fold3: mtry=29 
## + Fold4: mtry= 2 
## - Fold4: mtry= 2 
## + Fold4: mtry=15 
## - Fold4: mtry=15 
## + Fold4: mtry=29 
## - Fold4: mtry=29 
## Aggregating results
## Selecting tuning parameters
## Fitting mtry = 2 on full training set
```

```r
#modelintraininginFitbitTrain2

#str(valuesintraininginFitbitTrain1)
valuesintestinginFitbitTrain1 <-predict(preProc,testinginFitbitTrain1)

confusionMatrix(testinginFitbitTrain2$classe, predict(modelintraininginFitbitTrain2,valuesintestinginFitbitTrain1))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1392    2    0    0    1
##          B   12  930    7    0    0
##          C    0   11  840    4    0
##          D    0    1   19  783    1
##          E    0    0    0    5  896
## 
## Overall Statistics
##                                        
##                Accuracy : 0.987        
##                  95% CI : (0.984, 0.99)
##     No Information Rate : 0.286        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.984        
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.991    0.985    0.970    0.989    0.998
## Specificity             0.999    0.995    0.996    0.995    0.999
## Pos Pred Value          0.998    0.980    0.982    0.974    0.994
## Neg Pred Value          0.997    0.996    0.994    0.998    1.000
## Prevalence              0.286    0.192    0.177    0.162    0.183
## Detection Rate          0.284    0.190    0.171    0.160    0.183
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.995    0.990    0.983    0.992    0.998
```

```r
#################Now predicting the outcome of the 20 cases given in the assignment

fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(fileUrl, destfile="pml-testing.csv", method="curl")
FitbitTest<-read.csv("pml-testing.csv")
FitbitTest1<-FitbitTest[,-invalid]
valuesinFitbitTest1 <-predict(preProc,FitbitTest1)

answers <- predict(modelintraininginFitbitTrain2,valuesinFitbitTest1)
#predict(modelintraininginFitbitTrain2,valuesintestinginFitbitTrain1)
answers
```

```
##  [1] B A A A A B A B A A A A B A B B A B B B
## Levels: A B C D E
```
