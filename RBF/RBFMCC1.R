#Libraries and context
library(cluster)
library("MLmetrics")
library(mccr)
source("GowerFunction.R")
library("xgboost")  # the main algorithm
library("archdata") # for the sample dataset
library("caret")    # for the confusionmatrix() function (also needs e1071 package)
library("dplyr")    # for some data preperation
library("Ckmeans.1d.dp") # for xgb.ggplot.importance


setwd("/home/fran/TFG/RBF")

#Dataset preparation

df <- read.table("Frogs_MFCCs.csv", header=T, sep=",")

# ------- Procesado extra para testear ------------
df <- df[,-26]
names(df)[names(df) == "Family"] <- "target"
df$target <- as.integer(df[,"target"]) - 1
# ------------- How much neurons? -----------------
dims <- dim(df)

#Approach 1 -> Poisson lambda fixada
n <- rpois(1,dims[1]*0.05)
n

# ---------------- Which neurons? -----------------
Dissims <- GowerRBF(df) #Es una matriu triangular!!
Dissims[lower.tri(Dissims)] <- t(Dissims)[lower.tri(Dissims)] #Per completar la matriu de distancies
#Approach 1 -> distribució uniforme (random sampling)
chosen <- sample(1:dims[1],n)
chosen

# -------------- Data transformation --------------
#Approach 1 -> distribució uniforme
rbfdata <- Dissims[1:dims[1],chosen] #Aquí estàn guardades les dissimilituds de gower entre
#totes les dades i els centroides.

#----------------------- Sigmoid Hiperparameter ---------------------------

#Approach 1 -> Gower mean distance between random centroids
distclust <- Dissims[chosen,chosen]

x <- NULL

for (i in 1:n)
{
  sum <- 0
  for(j in distclust[i,])
  {
    sum <- sum+j
  }
  x <- append(x,sum/n)
}

length(x)
#--------------------------- Neural Network ------------------------------- 

# Test approach
rbfdatasig <- rbfdata

for (i in 1:n)
{
  rbfdatasig[,i] <- -rbfdatasig[,i]/x[i]
}

rbfdatasig <- 1/(1 + exp(rbfdatasig))

rbfdatasig <- data.frame(rbfdatasig,target = df[,"target"])

#Approach 1 -> Test with random centroids

smp_size <- floor(0.75 * dims[1])

train_ind <- sample(seq_len(dims[1]), size = smp_size)

data_variables <- as.matrix(rbfdatasig[,-(n+1)])
train <- data_variables[train_ind,]
test <- data_variables[-train_ind,]

trainpred <- rbfdatasig[train_ind,"target"]
testpred <- rbfdatasig[-train_ind, "target"]

train_matrix <- xgb.DMatrix(data = train, label = trainpred)
test_matrix <- xgb.DMatrix(data = test, label = testpred)

#Hyperparameter tuning
numberOfClasses <- length(unique(df$target))
xgb_params <- list("objective" = "multi:softprob", "eval_metric" = "mlogloss", "num_class" = numberOfClasses)
nround <- 50
cv.nfold <- 5

cv_model <- xgb.cv(params = xgb_params, data = train_matrix, nrounds = nround, nfold = cv.nfold, verbose = FALSE, prediction = TRUE)

pred <- data.frame(cv_model$pred)
predclasses <- max.col(pred,ties.method = "last")
OOF_prediction <- data.frame(cv_model$pred) %>% mutate(max_prob = max.col(., ties.method = "last"), label = trainpred + 1)

# confusion matrix
confusionMatrix(factor(OOF_prediction$max_prob), factor(OOF_prediction$label), mode = "everything")

#Real model 
#At this point the hiperparameters have been chosen. Now is time to fit the model.

bst_model <- xgb.train(params = xgb_params, data = train_matrix, nrounds = nround)

# Predict hold-out test set
test_pred <- predict(bst_model, newdata = test_matrix)
test_prediction <- matrix(test_pred, nrow = numberOfClasses, ncol=length(test_pred)/numberOfClasses) %>% t() %>% data.frame() %>% mutate(label = testpred + 1, max_prob = max.col(., "last"))
# confusion matrix of test set
confusionMatrix(factor(test_prediction$max_prob), factor(test_prediction$label), mode = "everything")

#Metrics for F1Score and Mathew's Correlation Coeff Score
Truepositives <- c()
Falsepositives <- c()
Truenegatives <- c()
Falsenegatives <- c()
for (i in 1:numberOfClasses)
{
  Truepositives <- append(Truepositives,0)
  Falsepositives <- append(Falsepositives,0)
  Truenegatives <- append(Truenegatives,0)
  Falsenegatives <- append(Falsenegatives,0)
}

for(i in 1:length(test_prediction[,"label"]))
{
  predclass <- test_prediction[i,"max_prob"]
  trueclass <- test_prediction[i,"label"]
  if(predclass == trueclass)
  {
    Truepositives[predclass] <- Truepositives[predclass] + 1
    for(j in 1:numberOfClasses)
    {
      if (j!= predclass)
      {
        Truenegatives[j] <- Truenegatives[j] + 1
      }
    }
  }
  if(predclass != trueclass)
  {
    Falsepositives[predclass] <- Falsepositives[predclass] + 1
    Falsenegatives[trueclass] <- Falsenegatives[trueclass] + 1
  }
}

#Micro F1-score

PMicro <- sum(Truepositives)/sum(Truepositives + Falsepositives)
RMicro <- sum(Truepositives)/sum(Truepositives + Falsenegatives)

F1Micro <- 2*(PMicro * RMicro)/(PMicro + RMicro)

#Macro F1-score

PMacro <- (1/numberOfClasses) * sum(Truepositives/(Truepositives+Falsepositives))
RMacro <- (1/numberOfClasses) * sum(Truepositives/(Truepositives+Falsenegatives))

F1Macro <- 2*(PMacro * RMacro)/(PMacro + RMacro)

# Mathew's score multiclass
TP <- sum(Truepositives)
TN <- sum(Truenegatives)
FP <- sum(Falsepositives)
FN <- sum(Falsenegatives)

MCC <- (TP*TN - FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))







