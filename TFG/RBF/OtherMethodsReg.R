#Libraries and context
library("dplyr") 
library(cluster)
library("MLmetrics")
library(mccr)
library(FNN)
source("GowerFunction.R")

setwd("/home/fran/TFG/RBF")

#Dataset preparation

df <- read.table("energydata_complete.csv", header=T, sep=",")

# ------- Procesado extra para testear ------------
df <- df[,-1]
names(df)[names(df) == "T_out"] <- "target"
df <- df[sample(nrow(df), 5000),]
# ------------- How much neurons? -----------------
dims <- dim(df)

#------------ Comparison with standard regression --------------

smp_size <- floor(0.75 * dims[1])

train_ind <- sample(seq_len(dims[1]), size = smp_size)

train3 <- df[train_ind, ]
test3 <- df[-train_ind,]

trainpred3 <- df[train_ind,"target"]
testpred3 <- df[-train_ind,"target"]

vars <- colnames(df)
vars <- vars[-21]

formulastring3 <- "target ~ "
for (i in 1:length(vars))
{
  if(i != length(vars)) strformula <- sprintf("%s + ",vars[i])
  else strformula <- sprintf("%s",vars[i])
  formulastring3 <- paste(formulastring3,strformula,sep="")
}

model3 <- lm(formula = as.formula(formulastring3), data = train3,family = gaussian)

summary(model3)

predictionsTrain3 <- predict(model3,train3)
predictionsTest3 <- predict(model3,test3)
predictionsTotal3 <- predict(model3,df)

#Error %
norm2 <- function(x) sqrt(sum(x^2))

NMSETrain3 <- norm2(train3[,"target"]-predictionsTrain3)/norm2(train3[,"target"])
NMSETest3 <- norm2(test3[,"target"]-predictionsTest3)/norm2(test3[,"target"])
NMSETotal3 <- norm2(df[,"target"]-predictionsTotal3)/norm2(df[,"target"])

#------------------------ Comparison with KNN ----------------------

smp_size <- floor(0.75 * dims[1])

train_ind <- sample(seq_len(dims[1]), size = smp_size)

train4 <- df[train_ind, -21 ]
test4 <- df[-train_ind, -21 ]

trainpred4<- df[train_ind,"target"]
testpred4 <- df[-train_ind,"target"]

model4 <- knn.reg(train=train4,test = test4,y=trainpred4, k = 10)

predictionsTrain4 <- knn.reg(train=train4,test = train4,y=trainpred4, k = 10)
predictionsTest4 <- knn.reg(train=train4,test = test4,y=trainpred4, k = 10)
predictionsTotal4 <- knn.reg(train=train4,test = df[,-21],y=trainpred4, k = 10)

predictionsTrain4 <- predictionsTrain4$pred
predictionsTest4 <- predictionsTest4$pred
predictionsTotal4 <- predictionsTotal4$pred

#Error %
norm2 <- function(x) sqrt(sum(x^2))

NMSETrain4 <- norm2(trainpred4-predictionsTrain4)/norm2(trainpred4)
NMSETest4 <- norm2(testpred4-predictionsTest4)/norm2(testpred4)
NMSETotal4 <- norm2(df[,"target"]-predictionsTotal4)/norm2(df[,"target"])


#-------------------- Comparison with normal RBF -------------------
#Approach 1 -> Poisson lambda fixada
#mateixa n que a cas 1, si no escollim una nova
#n <- rpois(1,dims[1]*0.05)
#n

#Approach 1 -> distribució uniforme (random sampling)
chosen <- sample(1:dims[1],n)
chosen

# Distance calculation
distancematrix <- dist(df,method="euclidean")

eucdistMatrix <- matrix(0,dims[1],dims[1])
eucdistMatrix[lower.tri(eucdistMatrix,diag=FALSE)] <- distancematrix 
eucdistMatrix[upper.tri(eucdistMatrix)] <- t(eucdistMatrix)[upper.tri(eucdistMatrix)]

# Using n and chosen vector like case 1 of our RBF
# Data transformation to RBF

rbfnormaldata <- eucdistMatrix[1:dims[1],chosen]

# Gaussian aplication

rbfnormdata <- rbfnormaldata

for(i in 1:n)
{
  variance = var(rbfnormaldata[,i])
  rbfnormdata[,i] <- (-(rbfnormdata[,i])^2)/(2*variance)
}

rbfnormdata <- exp(rbfnormdata)

rbfnormdata <- data.frame(rbfnormdata,target = df[,"target"])

#mete la transformacion en rbfnormdata

# Neural Network
#Approach 1 -> Test with random centroids

smp_size <- floor(0.75 * dims[1])

train_ind <- sample(seq_len(dims[1]), size = smp_size)

train5 <- rbfnormdata[train_ind, ]
test5 <- rbfnormdata[-train_ind,]

trainpred5 <- df[train_ind,"target"]
testpred5 <- df[-train_ind, "target"]

formulastring3 <- "target ~ "
for (i in 1:n)
{
  if(i != n) strformula <- sprintf("X%d + ",i)
  else strformula <- sprintf("X%d",i)
  formulastring3 <- paste(formulastring3,strformula,sep="")
}

model5 <- lm(formula = as.formula(formulastring3), data = train5, family = binomial)

summary(model5)

predictionsTrain5 <- predict(model5,train5)
predictionsTest5 <- predict(model5,test5)
predictionsTotal5 <- predict(model5,rbfnormdata)

#Error %
norm2 <- function(x) sqrt(sum(x^2))

NMSETrain5 <- norm2(train5[,"target"]-predictionsTrain5)/norm2(train5[,"target"])
NMSETest5 <- norm2(test5[,"target"]-predictionsTest5)/norm2(test5[,"target"])
NMSETotal5 <- norm2(rbfnormdata[,"target"]-predictionsTotal5)/norm2(rbfnormdata[,"target"])


