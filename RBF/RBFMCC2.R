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

#Approach -> Poisson lambda depén de la quantitat màxima de dades
m <- rpois(1,dims[1]*0.05)
m

# ---------------- Which neurons? -----------------
#Approach 2 -> picking cluster centroids.
#clustering
dissimMatrix <- daisy(df, metric = "gower", stand=TRUE)
distMatrix <- dissimMatrix^2
h1 <- hclust(distMatrix,method="ward.D2")
c2 <- cutree(h1,m)

dd <- cbind(df,c2)

#getting the centroids

c0 <- colMeans(data.matrix(df[dd$c2 == 1, ]))

ds <- data.frame(c0)

for (i in 2:m)
{
  c0 <- colMeans(data.matrix(df[dd$c2 == i, ]))
  ds <- cbind(ds,c0)
}

#the centroids are on ds

#computar la media por cada variable y crear una row que se convierta en el centroide
# -------------- Data transformation --------------
#Approach 2 -> gower distance between centroids and the rest of the data
tds <- data.frame(t(ds))
numericdata <- data.frame(data.matrix(df))

types <- sapply(df,class)

for (i in 1:dims[2])
{
  if(types[i] == "integer")
  {
    tds[,i] <- sapply(tds[,i],as.integer)
    numericdata[,i] <- sapply(numericdata[,i],as.integer)
  }
  if(types[i] == "character")
  {
    tds[,i] <- trunc(tds[,i])
    tds[,i] <- sapply(tds[,i],as.character)
    numericdata[,i] <- sapply(numericdata[,i],as.character)
  }
  if(types[i] == "factor")
  {
    if(nlevels(df[,i]) == 2)
    {
      for(j in 1:n)
      {
        if(tds[j,i] < 1.5) tds[j,i] <- 1
        else tds[j,i] <- 2
      }
    }
    tds[,i] <- trunc(tds[,i])
    tds[,i] <- sapply(tds[,i],as.factor)
    numericdata[,i] <- sapply(numericdata[,i],as.factor)
  }
  if(types[i] == "logical")
  {
    for(j in 1:n)
    {
      if(tds[j,i]<0.5) tds[j,i] <- 0
      else tds[j,i] <- 1
    }
    tds[,i] <- sapply(tds[,i],as.logical)
    numericdata[,i] <- sapply(numericdata[,i],as.logical)
  }
}

fusedata <- rbind(numericdata,tds)

distances <- GowerRBF(fusedata)
distances[lower.tri(distances)] <- t(distances)[lower.tri(distances)]

extrasize <- c(1:m)
extrasize <- dims[1] + extrasize

rbfdata2 <- distances[1:dims[1],extrasize]

#----------------------- Sigmoid Hiperparameter ---------------------------
#Approach 2 -> Dispersion index from a cluster

y <- NULL

for (i in 1:m)
{
  dissimsi <- Dissims[dd$c2 == i , ]
  y <- append(y,max(dissimsi))
}

length(y)

#--------------------------- Neural Network ------------------------------- 

rbfdatasig2 <- rbfdata2
for (j in 1:m)
{
  rbfdatasig2[,j] <- -rbfdatasig2[,j]/y[j]
}

rbfdatasig2 <- 1/(1 + exp(rbfdatasig2))

rbfdatasig2 <- data.frame(rbfdatasig2,target = df[,"target"])

#Approach 2 -> Test with cluster centroids

smp_size <- floor(0.75 * dims[1])

train_ind <- sample(seq_len(dims[1]), size = smp_size)

data_variables2 <- as.matrix(rbfdatasig2[,-(m+1)])
train2 <- data_variables2[train_ind,]
test2 <- data_variables2[-train_ind,]

trainpred2 <- rbfdatasig2[train_ind,"target"]
testpred2 <- rbfdatasig2[-train_ind, "target"]

train_matrix2 <- xgb.DMatrix(data = train2, label = trainpred2)
test_matrix2 <- xgb.DMatrix(data = test2, label = testpred2)

#Hyperparameter tuning
numberOfClasses <- length(unique(df$target))
xgb_params <- list("objective" = "multi:softprob", "eval_metric" = "mlogloss", "num_class" = numberOfClasses)
nround2 <- 50
cv.nfold2 <- 5

cv_model2 <- xgb.cv(params = xgb_params, data = train_matrix2, nrounds = nround2, nfold = cv.nfold2, verbose = FALSE, prediction = TRUE)

pred2 <- data.frame(cv_model2$pred)
predclasses2 <- max.col(pred2,ties.method = "last")
OOF_prediction2 <- data.frame(cv_model2$pred) %>% mutate(max_prob = max.col(., ties.method = "last"), label = trainpred2 + 1)

# confusion matrix
confusionMatrix(factor(OOF_prediction2$max_prob), factor(OOF_prediction2$label), mode = "everything")

#Real model 
#At this point the hiperparameters have been chosen. Now is time to fit the model.

bst_model2 <- xgb.train(params = xgb_params, data = train_matrix2, nrounds = nround2)

# Predict hold-out test set
test_pred2 <- predict(bst_model2, newdata = test_matrix2)
test_prediction2 <- matrix(test_pred2, nrow = numberOfClasses, ncol=length(test_pred2)/numberOfClasses) %>% t() %>% data.frame() %>% mutate(label = testpred2 + 1, max_prob = max.col(., "last"))
# confusion matrix of test set
confusionMatrix(factor(test_prediction2$max_prob), factor(test_prediction2$label), mode = "everything")

#Metrics for F1Score and Mathew's Correlation Coeff Score
Truepositives2 <- c()
Falsepositives2 <- c()
Truenegatives2 <- c()
Falsenegatives2 <- c()
for (i in 1:numberOfClasses)
{
  Truepositives2 <- append(Truepositives2,0)
  Falsepositives2 <- append(Falsepositives2,0)
  Truenegatives2 <- append(Truenegatives2,0)
  Falsenegatives2 <- append(Falsenegatives2,0)
}

for(i in 1:length(test_prediction2[,"label"]))
{
  predclass <- test_prediction2[i,"max_prob"]
  trueclass <- test_prediction2[i,"label"]
  if(predclass == trueclass)
  {
    Truepositives2[predclass] <- Truepositives2[predclass] + 1
    for(j in 1:numberOfClasses)
    {
      if (j!= predclass)
      {
        Truenegatives2[j] <- Truenegatives2[j] + 1
      }
    }
  }
  if(predclass != trueclass)
  {
    Falsepositives2[predclass] <- Falsepositives2[predclass] + 1
    Falsenegatives2[trueclass] <- Falsenegatives2[trueclass] + 1
  }
}

#Micro F1-score

PMicro2 <- sum(Truepositives2)/sum(Truepositives2 + Falsepositives2)
RMicro2 <- sum(Truepositives2)/sum(Truepositives2 + Falsenegatives2)

F1Micro2 <- 2*(PMicro2 * RMicro2)/(PMicro2 + RMicro2)

#Macro F1-score

PMacro2 <- (1/numberOfClasses) * sum(Truepositives2/(Truepositives2+Falsepositives2))
RMacro2 <- (1/numberOfClasses) * sum(Truepositives2/(Truepositives2+Falsenegatives2))

F1Macro <- 2*(PMacro2 * RMacro2)/(PMacro2 + RMacro2)

# Mathew's score multiclass
TP2 <- sum(Truepositives2)
TN2 <- sum(Truenegatives2)
FP2 <- sum(Falsepositives2)
FN2 <- sum(Falsenegatives2)

MCC2 <- (TP2*TN2 - FP2*FN2)/sqrt((TP2+FP2)*(TP2+FN2)*(TN2+FP2)*(TN2+FN2))
