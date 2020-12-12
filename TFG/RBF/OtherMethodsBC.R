#Libraries and context
library("dplyr") 
library(cluster)
library("MLmetrics")
library(mccr)
source("GowerFunction.R")

setwd("/home/fran/TFG/RBF")

#Dataset preparation

df <- read.table("heart.csv", header=T, sep=",")

# ------- Procesado extra para testear ------------
#df <- df[-1]
#df[1,1] <- 20
# ------------- How much neurons? -----------------
dims <- dim(df)

#------------ Comparison with standard logistic regression --------------

smp_size <- floor(0.75 * dims[1])

train_ind <- sample(seq_len(dims[1]), size = smp_size)

train3 <- df[train_ind, ]
test3 <- df[-train_ind,-ncol(df)]

trainpred3 <- df[train_ind,"target"]
testpred3 <- df[-train_ind,"target"]

vars <- colnames(df[,-ncol(df)])
formulastring3 <- "target ~ "
for (i in 1:length(vars))
{
  if(i != length(vars)) strformula <- sprintf("%s + ",vars[i])
  else strformula <- sprintf("%s",vars[i])
  formulastring3 <- paste(formulastring3,strformula,sep="")
}

model3 <- glm(formula = as.formula(formulastring3), data = train3,family = binomial)

summary(model3)

probabilities3 <- predict(model2,data.frame(test2),type ="response")
predicted.classes3 <- ifelse(probabilities2 > 0.5, 1 , 0)

#Error %
error_count3 <- 0

for (i in 1:length(testpred3))
{
  if(predicted.classes3[i] != testpred3[i]) error_count3 <- error_count3 + 1
}

error_percent3 = error_count3 / length(testpred3)
error_percent3

# F1 Score

f1.3 <- F1_Score(testpred3,predicted.classes3)
f1.3

#Mathew's Score

mccr3 <- mccr(testpred3,predicted.classes3)
mccr3

#------------------------ Comparison with KNN ----------------------
k <- 5

predicted.classes4 <- NULL


for (i in 1:dims[1])
{
  dist <- order(Dissims[i,])
  dist <- dist[-1]
  
  kt <- df$target[dist[1:k]]
  n1s <- sum(kt[kt == 1])/length(kt)
  n0s <- sum(kt[kt == 0])/length(kt)
  
  if(n1s > n0s) predicted.classes4 <- append(predicted.classes4, 1)
  else predicted.classes4 <- append(predicted.classes4, 0)
}

predicted.classes4

error_count4 <- 0

for (i in 1:dims[1])
{
  if(predicted.classes4[i] != df$target[i]) error_count4 <- error_count4 + 1
}

error_percent4 = error_count4 / dims[1]
error_percent4

#F1 Score

f1.4 <- F1_Score(df$target,predicted.classes4)
f1.4

#Mathew's Correlation Coefficient
mccr4 <- mccr(df$target,predicted.classes4)
mccr4 

#-------------------- Comparison with normal RBF -------------------
#Approach 1 -> Poisson lambda fixada
n <- rpois(1,20)
n

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
test5 <- rbfnormdata[-train_ind,-(n+1)]

trainpred5 <- df[train_ind,"target"]
testpred5 <- df[-train_ind, "target"]

formulastring3 <- "target ~ "
for (i in 1:n)
{
  if(i != n) strformula <- sprintf("X%d + ",i)
  else strformula <- sprintf("X%d",i)
  formulastring3 <- paste(formulastring3,strformula,sep="")
}

model5 <- glm(formula = as.formula(formulastring3), data = train5, family = binomial)

summary(model5)

probabilities5 <- predict(model5,data.frame(test5),type ="response")
probabilities5
predicted.classes5 <- ifelse(probabilities5 > 0.5, 1 , 0)

#Error %
error_count5 <- 0

for (i in 1:length(testpred5))
{
  if(predicted.classes5[i] != testpred5[i]) error_count5 <- error_count5 + 1
}

error_percent5 = error_count5 / length(testpred5)
print(error_percent5)

#F1 Score

f1.5 <- F1_Score(testpred5,predicted.classes5)
f1.5

#Mathew's Score

mccr5 <- mccr(testpred5,predicted.classes5)
mccr5

