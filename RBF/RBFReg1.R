#Libraries and context
library("dplyr") 
library(cluster)
library("MLmetrics")
library(mccr)
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

train <- rbfdatasig[train_ind, ]
test <- rbfdatasig[-train_ind,]

trainpred <- rbfdatasig[train_ind,"target"]
testpred <- rbfdatasig[-train_ind, "target"]

formulastring <- "target ~ "
for (i in 1:n)
{
  if(i != n) strformula <- sprintf("X%d + ",i)
  else strformula <- sprintf("X%d",i)
  formulastring <- paste(formulastring,strformula,sep="")
}

model <- lm(formula = as.formula(formulastring), data = train, family = gaussian)

summary(model)

predictionsTrain <- predict(model,train)
predictionsTest <- predict(model,test)
predictionsTotal <- predict(model,rbfdatasig)

#Error %
#NMSE
norm2 <- function(x) sqrt(sum(x^2))

NMSETrain <- norm2(train[,"target"]-predictionsTrain)/norm2(train[,"target"])
NMSETest <- norm2(test[,"target"]-predictionsTest)/norm2(test[,"target"])
NMSETotal <- norm2(rbfdatasig[,"target"]-predictionsTotal)/norm2(rbfdatasig[,"target"])

